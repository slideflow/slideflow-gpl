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

import sys
import pkgutil

def register_extras():
    # Register the additional pretrained feature extractors
    from . import extractors
    for submodule in pkgutil.iter_modules(extractors.__path__):
        module = submodule.module_finder.find_spec(submodule.name).loader.load_module(submodule.name)
        sys.modules[f'slideflow.model.extractors.{submodule.name}'] = module

    # Register CLAM
    from . import clam
    sys.modules['slideflow.clam'] = clam