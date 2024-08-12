# Slideflow-GPL - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-GPL.
#
# Slideflow-GPL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-GPL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-GPL. If not, see <https://www.gnu.org/licenses/>.

from slideflow.model.extractors import register_torch

@register_torch
def histossl(**kwargs):
    from .histossl import HistoSSLFeatures
    return HistoSSLFeatures(**kwargs)

@register_torch
def ctranspath(**kwargs):
    from .ctranspath import CTransPathFeatures
    return CTransPathFeatures(**kwargs)

@register_torch
def retccl(**kwargs):
    from .retccl import RetCCLFeatures
    return RetCCLFeatures(**kwargs)
