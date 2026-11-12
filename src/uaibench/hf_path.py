"""HuggingFace transformers setup for MBGen (transformers 4.37.2 from the environment)."""


def add():
    import transformers.utils as tu
    import transformers.utils.import_utils as iu
    off = lambda: False  # T5 needs no vision code; skip optional vision imports
    for m in (tu, iu):
        m.is_vision_available = off
        m.is_torchvision_available = off
