# GSABeam

**Grammar-State Aware Beam Search** for enhancing structural diversity in constrained LLM generation.

Implements the decoding strategy from the paper *Grammar-State Aware Beam Search for Enhancing Structural Diversity in LLM Generation*.

## Overview

GSA-Beam reorganizes beam search around grammar states rather than lexical prefixes, enabling:

1. **State-centric beam organization**: Group hypotheses by grammar state \(B_t^{(s)}\) for explicit structural diversification
2. **Dynamic state-level beam regulation**: Adapt beam width \(K_{\text{dynamic}}\) based on model uncertainty and structural branching
3. **Higher throughput**: Up to 2.3× higher TPS than conventional constrained beam search (see paper)

## Repository Structure

```
GSABeam/
├── README.md
├── DESIGN.md
├── gsa_beam.py
├── .gitignore
└── integration/
    ├── base_grammar_backend_patch.py
    ├── gram2token_backend_patch.py
    ├── sampling_params_patch.py
    └── sampling_batch_info_patch.py
```

## Installation

No standalone installation. Integrate into SGLang as described below.

## SGLang Integration

### 1. Copy core module

```bash
cp GSABeam/gsa_beam.py /path/to/sglang/python/sglang/srt/constrained/
```

### 2. Apply integration patches

The `integration/` folder contains the modifications needed for SGLang:

- **base_grammar_backend.py**: Add `GSABeamGrammarMixin` and optional `get_grammar_state()` interface
- **gram2token_backend.py**: Implement `get_grammar_state()` in `Gram2TokenGrammar`
- **sampling_params.py**: Add `gsa_beam`, `gsa_beam_K_min`, `gsa_beam_K_max`
- **sampling_batch_info.py**: Add `use_gsa_beam` and related fields in `from_schedule_batch()`

See `integration/` for full patch contents.

### 3. Use GSA-Beam

When using constrained decoding with multiple candidates:

```python
# API request
{
    "text": "Generate user profile",
    "sampling_params": {
        "n": 16,
        "json_schema": schema,
        "gsa_beam": True,
        "gsa_beam_K_max": 1024
    }
}
```

Enable when: `grammar` + `n > 1` + `gsa_beam=True`.

## Requirements

- SGLang
- PyTorch
- Grammar backend with state support (e.g., gram2token; xgrammar may need extension)

## References

- Paper: *Grammar-State Aware Beam Search for Enhancing Structural Diversity in LLM Generation*
- SGLang: high-performance LLM inference framework
- XGrammar: grammar-constrained decoding backend

## License

Apache 2.0 (aligned with SGLang)
