# Add to base_grammar_backend.py (before class BaseGrammarObject)

class GSABeamGrammarMixin:
    """
    Optional mixin for grammar objects supporting GSA-Beam.

    When implemented, enables Grammar-State Aware Beam Search for
    structurally diverse constrained multi-candidate generation.
    """

    def get_grammar_state(self, idx: int = 0) -> int:
        """
        Return the current grammar state ID for this hypothesis.

        Override in backends that support GSA-Beam (e.g., Gram2Token).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support GSA-Beam."
        )
