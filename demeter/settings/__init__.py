import contextlib

from .settings import *

with contextlib.suppress(ImportError):
    from .settings_overrides impoer *
