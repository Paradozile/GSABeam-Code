# 1. Update class definition:
#    class Gram2TokenGrammar(GSABeamGrammarMixin, BaseGrammarObject):

# 2. Add import:
#    from sglang.srt.constrained.base_grammar_backend import GSABeamGrammarMixin

# 3. Add method to Gram2TokenGrammar:

    def get_grammar_state(self, idx: int = 0) -> int:
        """Return current grammar state for GSA-Beam support."""
        if self.state_ids.dim() == 0:
            return int(self.state_ids.item())
        return int(self.state_ids[idx].item())
