from .config import CLAMModelConfig, CLAMTrainerConfig

from slideflow.mil import register_trainer, register_model


@register_trainer
def clam():
    return CLAMTrainerConfig

@register_model
def clam_sb():
    from .model import CLAM_SB
    return CLAM_SB

@register_model
def clam_mb():
    from .model import CLAM_MB
    return CLAM_MB

@register_model
def mil_fc():
    from .model import MIL_fc
    return MIL_fc

@register_model
def mil_fc_mc():
    from .model import MIL_fc_mc
    return MIL_fc_mc
