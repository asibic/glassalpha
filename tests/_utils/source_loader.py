import inspect
from importlib import resources
from types import ModuleType
from typing import Optional


def get_module_source(
    module: ModuleType,
    pkg_fallback: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Return source code for a loaded module, working for both source checkouts and installed wheels.
    Tries inspect.getsource first; falls back to importlib.resources if needed.
    """
    # Primary: inspect.getsource (works when .py is available)
    try:
        return inspect.getsource(module)
    except OSError:
        pass

    # Fallback: importlib.resources to read the file inside the dist
    if pkg_fallback and filename:
        return (resources.files(pkg_fallback) / filename).read_text(encoding="utf-8")

    # If still unavailable, raise a helpful error
    raise FileNotFoundError(f"Could not load source for module {module.__name__}")
