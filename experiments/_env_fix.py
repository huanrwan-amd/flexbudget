"""Environment shim for the rocm/pytorch:...pytorch_2.10.0 container.

The container's torchvision is built against a different torch and crashes on import
(`RuntimeError: operator torchvision::nms does not exist` -> partially-initialized module).
transformers (>=4.48) imports torchvision via its image/loss utilities on model load, which
breaks all AutoModel loads even for pure-text LLMs.

We only use text models, so we tell transformers that torchvision is UNAVAILABLE. Its
vision imports are guarded by `is_torchvision_available()`, so forcing that False makes
transformers skip torchvision entirely -- without polluting sys.modules (a fake torchvision
module leaks placeholder objects into torch and breaks flex_attention).

Import this module FIRST, before importing transformers. Idempotent and safe if torchvision
actually works (then it does nothing).
"""
from __future__ import annotations

import sys
import types
import importlib.machinery


def _torchvision_ok() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


class _Dummy:
    """Attribute-transparent placeholder (dunder-guarded to avoid breaking introspection)."""
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _DUMMY

    def __call__(self, *_a, **_k):
        return _DUMMY

    def __getitem__(self, _k):
        return _DUMMY


_DUMMY = _Dummy()


def stub_torchaudio() -> str:
    """torchaudio's native lib (libtorchaudio.so) is broken in the container, and transformers
    imports it UNCONDITIONALLY via loss_utils -> loss_rnnt. Availability flags do not help, so
    install a minimal dunder-guarded stub in sys.modules before transformers loads. torch does
    not import torchaudio, so this does not affect flex_attention."""
    try:
        import torchaudio  # noqa: F401
        return "real"
    except Exception:
        pass
    for name in list(sys.modules):
        if name == "torchaudio" or name.startswith("torchaudio."):
            sys.modules.pop(name, None)
    for name in ("torchaudio", "torchaudio.functional", "torchaudio.transforms",
                 "torchaudio.models"):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []

        def __getattr__(attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _DUMMY

        m.__dict__["__getattr__"] = __getattr__
        sys.modules[name] = m
    sys.modules["torchaudio"].__version__ = "0.0.0+stub"
    return "stub"


def disable_torchvision_in_transformers() -> str:
    if _torchvision_ok():
        return "real"  # torchvision fine (e.g. login node); leave everything alone.

    def _false() -> bool:
        return False

    patched = []
    # import_utils holds the cached flag + the canonical predicate. Importing it does NOT
    # import torchvision, so this is safe. Set BEFORE any transformers vision module loads.
    try:
        import transformers.utils.import_utils as iu
        iu._torchvision_available = False
        iu.is_torchvision_available = _false
        patched.append("import_utils")
    except Exception:
        pass
    # transformers.utils re-exports the predicate; importing it is also torchvision-free.
    # Do NOT import image_utils / image_transforms here -- those trigger the crash.
    try:
        import transformers.utils as u
        if hasattr(u, "is_torchvision_available"):
            u.is_torchvision_available = _false
        patched.append("utils")
    except Exception:
        pass
    return "disabled:" + ",".join(patched)


STATUS = disable_torchvision_in_transformers()
STATUS_AUDIO = stub_torchaudio()
