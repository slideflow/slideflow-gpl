import sys
import pkgutil

def register_extras():
    # Register the additional pretrained feature extractors
    from . import extractors
    for submodule in pkgutil.iter_modules(extractors.__path__):
        module = submodule.module_finder.find_spec(submodule.name).loader.load_module(submodule.name)
        sys.modules[f'slideflow.model.extractors.{submodule.name}'] = module

    # Register BISCUIT
    from . import biscuit
    sys.modules['slideflow.biscuit'] = biscuit

    # Register CLAM
    from . import clam