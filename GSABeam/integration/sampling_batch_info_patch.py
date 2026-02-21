# Add to SamplingBatchInfo (class fields):

    # GSA-Beam: Grammar-State Aware Beam Search
    use_gsa_beam: bool = False
    gsa_beam_n_target: Optional[int] = None
    gsa_beam_K_min: Optional[int] = None
    gsa_beam_K_max: int = 1024

# Add to from_schedule_batch(), before ret = cls(...):

        # GSA-Beam: enable when grammar + n>1 + gsa_beam=True
        use_gsa_beam = False
        gsa_beam_n_target = None
        gsa_beam_K_min = None
        gsa_beam_K_max = 1024
        if batch.has_grammar and reqs:
            sp0 = reqs[0].sampling_params
            n_val = getattr(sp0, "n", 1)
            if n_val > 1 and getattr(sp0, "gsa_beam", False):
                use_gsa_beam = True
                gsa_beam_n_target = n_val
                gsa_beam_K_min = getattr(sp0, "gsa_beam_K_min", None) or n_val
                gsa_beam_K_max = getattr(sp0, "gsa_beam_K_max", 1024)

# Add to cls(...) call: use_gsa_beam=..., gsa_beam_n_target=..., etc.
