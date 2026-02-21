# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Grammar-State Aware Beam Search (GSA-Beam).

Implements the decoding strategy from:
  "Grammar-State Aware Beam Search for Enhancing Structural Diversity in LLM Generation"

Key components:
  1. State-centric beam organization: Group hypotheses by grammar state B_t^(s)
  2. Dynamic state-level beam regulation: K_dynamic based on entropy and branching
  3. State-level budget allocation: k_s = min(|B_t^(s)|, floor(B / |active_states|))
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)


@dataclass
class GSABeamHypothesis:
    """
    A single hypothesis in the GSA-Beam search.

    Attributes:
        output_ids: Generated token sequence y_{1:t}
        logprob: Accumulated log-probability log p(y_{1:t} | x)
        grammar_state: Current constraint state s(y_{1:t})
        grammar_obj: Reference to grammar object for state tracking
        req_idx: Optional index mapping to original request
    """

    output_ids: List[int]
    logprob: float
    grammar_state: int
    grammar_obj: Any
    req_idx: Optional[int] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _compute_entropy(probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute entropy H(P) = -sum p_i log p_i over legal tokens.

    Args:
        probs: Token probability distribution [vocab_size]
        mask: Optional boolean mask of legal tokens (True = legal)

    Returns:
        Entropy value, or 0.0 if no valid tokens.
    """
    if mask is not None:
        probs = probs * mask.float()
    probs = probs.clamp(min=1e-10)
    probs = probs / probs.sum()
    entropy = -(probs * torch.log(probs)).sum().item()
    return max(0.0, entropy)


def compute_K_dynamic(
    n: int,
    H_P: float,
    A_max: int,
    H_max: float,
    K_min: int,
    K_max: int,
) -> int:
    """
    Compute dynamic beam width per paper Eq. 3-4.

    K_dynamic = clip(ceil(n * (H(P) + |A_max|) / (H_max + A_max)), K_min, K_max)

    Args:
        n: Target number of output candidates
        H_P: Entropy of valid next-token distribution
        A_max: max_{s in S_t} |A(s)|
        H_max: log |V_legal|
        K_min: Minimum beam width (typically n)
        K_max: Maximum beam width (e.g., 1024)

    Returns:
        Dynamic beam width for current step
    """
    if H_max <= 0 or (H_max + A_max) <= 0:
        return max(K_min, min(K_max, n))
    raw = math.ceil(n * (H_P + A_max) / (H_max + A_max))
    return max(K_min, min(K_max, int(raw)))


def allocate_state_budgets(
    K_total: int,
    state_groups: Dict[int, List[GSABeamHypothesis]],
) -> Dict[int, int]:
    """
    Allocate beam budget per active grammar state (paper Eq. 2).

    k_s = min(|B_t^(s)|, floor(B / |{s : B_t^(s) != empty}|))

    Args:
        K_total: Total beam width (K_dynamic)
        state_groups: {state_id: [hypotheses]}

    Returns:
        {state_id: budget} for each active state
    """
    active_states = [s for s, hyps in state_groups.items() if hyps]
    num_active = len(active_states)
    if num_active == 0:
        return {}
    base_budget = K_total // num_active
    budgets = {}
    for s in active_states:
        count = len(state_groups[s])
        budgets[s] = min(count, base_budget)
    return budgets


def group_by_successor_state(
    hypotheses: List[GSABeamHypothesis],
    grammar_backend: Any = None,
) -> Dict[int, List[GSABeamHypothesis]]:
    """
    Group candidate hypotheses by their successor grammar state.

    Args:
        hypotheses: List of hypotheses to group
        grammar_backend: Unused; for API compatibility

    Returns:
        {state_id: [hypotheses]}
    """
    groups: Dict[int, List[GSABeamHypothesis]] = defaultdict(list)
    for h in hypotheses:
        groups[h.grammar_state].append(h)
    return dict(groups)


def prune_beam_by_state(
    candidates_by_state: Dict[int, List[GSABeamHypothesis]],
    budgets: Dict[int, int],
) -> List[GSABeamHypothesis]:
    """
    Within each state, retain top-K^(s) hypotheses by log-probability.

    Args:
        candidates_by_state: {state_id: [candidate hypotheses]}
        budgets: {state_id: K^(s)}

    Returns:
        Pruned beam B_{t+1}
    """
    result = []
    for state_id, candidates in candidates_by_state.items():
        k = budgets.get(state_id, 0)
        if k <= 0 or not candidates:
            continue
        sorted_candidates = sorted(candidates, key=lambda h: h.logprob, reverse=True)
        result.extend(sorted_candidates[:k])
    return result


class GSABeamProcessor:
    """
    Processor for Grammar-State Aware Beam Search.

    Integrates with SGLang's constrained decoding pipeline.
    """

    def __init__(
        self,
        n_target: int,
        K_min: Optional[int] = None,
        K_max: int = 1024,
        vocab_size: int = 128256,
        device: str = "cuda",
    ):
        self.n_target = n_target
        self.K_min = K_min if K_min is not None else n_target
        self.K_max = K_max
        self.vocab_size = vocab_size
        self.device = device

    def compute_dynamic_beam_width(
        self,
        active_states: Set[int],
        probs_by_state: Optional[Dict[int, torch.Tensor]] = None,
        A_max: Optional[int] = None,
    ) -> int:
        """Compute K_dynamic for the current step."""
        if A_max is None:
            A_max = self.vocab_size
        V_legal = self.vocab_size
        H_max = math.log(max(1, V_legal))
        H_P = 0.0
        if probs_by_state and len(probs_by_state) > 0:
            all_probs = list(probs_by_state.values())[0]
            H_P = _compute_entropy(all_probs)
        return compute_K_dynamic(
            n=self.n_target,
            H_P=H_P,
            A_max=A_max,
            H_max=H_max,
            K_min=self.K_min,
            K_max=self.K_max,
        )

    def process_candidates(
        self,
        candidates: List[GSABeamHypothesis],
        K_dynamic: Optional[int] = None,
    ) -> List[GSABeamHypothesis]:
        """Group candidates by state, allocate budgets, prune."""
        state_groups = group_by_successor_state(candidates, None)
        if not state_groups:
            return []

        if K_dynamic is None:
            K_dynamic = self.compute_dynamic_beam_width(
                active_states=set(state_groups.keys()),
            )
        budgets = allocate_state_budgets(K_dynamic, state_groups)
        return prune_beam_by_state(state_groups, budgets)


def supports_gsa_beam(grammar_obj: Any) -> bool:
    """Check if the grammar object supports GSA-Beam (exposes grammar state)."""
    return hasattr(grammar_obj, "state_ids") or hasattr(
        grammar_obj, "get_grammar_state"
    )
