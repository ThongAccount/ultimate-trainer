import sys, os, importlib.util

_parent = os.path.dirname(os.path.dirname(__file__))
_hdir = os.path.join(_parent, 'ultimate-trainer')

_mods = {}
for _name in ['config', 'bitlinear', 'subqsa', 'model', 'train']:
    _spec = importlib.util.spec_from_file_location(
        'ultimate_trainer.' + _name,
        os.path.join(_hdir, _name + '.py'))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['ultimate_trainer.' + _name] = _mod
    _spec.loader.exec_module(_mod)
    _mods[_name] = _mod

globals().update(_mods)
del _parent, _hdir, _name, _spec, _mod, _mods
