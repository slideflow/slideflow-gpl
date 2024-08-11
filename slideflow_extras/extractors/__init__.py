from slideflow.model.extractors import register_torch

@register_torch
def histossl(**kwargs):
    from .histossl import HistoSSLFeatures
    return HistoSSLFeatures(**kwargs)

@register_torch
def plip(**kwargs):
    from .plip import PLIPFeatures
    return PLIPFeatures(**kwargs)

@register_torch
def ctranspath(**kwargs):
    from .ctranspath import CTransPathFeatures
    return CTransPathFeatures(**kwargs)

@register_torch
def retccl(**kwargs):
    from .retccl import RetCCLFeatures
    return RetCCLFeatures(**kwargs)
