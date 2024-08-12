from .config import CLAMModelConfig, LegacyCLAMTrainerConfig

from slideflow.mil import register_trainer, register_model


@register_trainer
def legacy_clam():
    return LegacyCLAMTrainerConfig

@register_model(config=CLAMModelConfig)
def clam_sb():
    from .model import CLAM_SB
    return CLAM_SB

@register_model(config=CLAMModelConfig)
def clam_mb():
    from .model import CLAM_MB
    return CLAM_MB

@register_model(config=CLAMModelConfig)
def mil_fc():
    from .model import MIL_fc
    return MIL_fc

@register_model(config=CLAMModelConfig)
def mil_fc_mc():
    from .model import MIL_fc_mc
    return MIL_fc_mc
