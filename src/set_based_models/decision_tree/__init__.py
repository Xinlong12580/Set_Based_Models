"""Compatibility imports for the optional :mod:`set_based_edt` extension."""

from set_based_models import _load_edt

_extension = _load_edt()
__all__ = _extension.__all__
globals().update({name: getattr(_extension, name) for name in __all__})
