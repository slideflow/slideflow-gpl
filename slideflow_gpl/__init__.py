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
import types
import importlib
import pkgutil


def _register_lazy_alias(alias, target):
    """Expose ``target`` under the back-compat import path ``alias`` without
    importing it. The real module (and its heavy deps, e.g. torch) is loaded on
    first attribute access, after which ``alias`` resolves to the real module."""
    if alias in sys.modules:
        return

    class _LazyAlias(types.ModuleType):
        def __getattr__(self, attr):
            real = importlib.import_module(target)
            sys.modules[alias] = real
            return getattr(real, attr)

    sys.modules[alias] = _LazyAlias(alias)


def register_extras():
    """Register slideflow-gpl's feature extractors and CLAM MIL models.

    Invoked at ``import slideflow`` time via the 'slideflow.plugins' entry point,
    so it deliberately avoids importing heavy model code. Importing the
    subpackages only runs their registration decorators (``@register_torch`` /
    ``@register_model``), which store lightweight factory callables; each factory
    imports its heavy modules lazily, only when the extractor/model is built.
    """
    # Feature extractors: registers the `ctranspath` / `retccl` factories.
    from . import extractors

    # Back-compat direct-import paths (e.g. `slideflow.model.extractors.ctranspath`),
    # resolved lazily so torch is not pulled in at registration time.
    for submodule in pkgutil.iter_modules(extractors.__path__):
        _register_lazy_alias(
            f'slideflow.model.extractors.{submodule.name}',
            f'{extractors.__name__}.{submodule.name}',
        )

    # CLAM MIL models: importing the subpackage runs the @register_model
    # decorators. The torch model definitions are imported lazily by the
    # factories; this import only pulls in config (no torch).
    from . import clam
    sys.modules.setdefault('slideflow.clam', clam)