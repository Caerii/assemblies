"""One construction boundary for historical fixed-connectome experiments."""


def model_semantics_kwargs(model_semantics):
    """Preserve standalone trial compatibility while enforcing runner profiles."""
    return ({"model_semantics": model_semantics}
            if model_semantics is not None else {})


def explicit_brain_from_values(
    brain_type, *, p, seed, w_max, model_semantics=None,
):
    """Construct the fixed dense CPU model and check its recorded profile."""
    return brain_type(
        p=p,
        seed=seed,
        w_max=w_max,
        engine="numpy_explicit",
        norm_init=False,
        model_semantics=model_semantics,
    )


def explicit_brain(brain_type, config, seed, model_semantics=None):
    """Construct the recorded dense CPU model and make its profile executable."""
    return explicit_brain_from_values(
        brain_type, p=config.p, seed=seed, w_max=config.w_max,
        model_semantics=model_semantics,
    )
