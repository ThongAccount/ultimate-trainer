import sys, os, importlib.util

_parent = os.path.dirname(os.path.dirname(__file__))
_hdir = os.path.join(_parent, 'subqsa-trainer')

# Pre-load all modules into sys.modules BEFORE importing them
# This lets subqsa-trainer/model.py do `import subqsa_trainer.subqsa`
# without needing a proper package hierarchy
mods = {}
for _name in ['config', 'subqsa', 'model', 'train']:
    _spec = importlib.util.spec_from_file_location(
        'subqsa_trainer.' + _name, os.path.join(_hdir, _name + '.py'))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['subqsa_trainer.' + _name] = _mod
    _spec.loader.exec_module(_mod)
    mods[_name] = _mod

# Now re-export for convenience
globals().update(mods)
del _parent, _hdir, _name, _spec, _mod, mods
