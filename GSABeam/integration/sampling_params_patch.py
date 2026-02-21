# Add to SamplingParams.__init__ (parameters):

        # GSA-Beam: Grammar-State Aware Beam Search
        gsa_beam: bool = False,
        gsa_beam_K_min: Optional[int] = None,
        gsa_beam_K_max: int = 1024,

# Add to SamplingParams.__init__ (body, after custom_params):

        # GSA-Beam parameters (used when gsa_beam=True and n>1 with grammar)
        self.gsa_beam = gsa_beam
        self.gsa_beam_K_min = gsa_beam_K_min  # None means use n
        self.gsa_beam_K_max = gsa_beam_K_max
