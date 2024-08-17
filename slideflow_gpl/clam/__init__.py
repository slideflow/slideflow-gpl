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

from .config import CLAMModelConfig, LegacyCLAMTrainerConfig

from slideflow.mil import register_model


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
