from __future__ import annotations

import importlib
from types import ModuleType


class MissingOptionalDependency(RuntimeError):
    pass


def optional_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def require_optional(module_name: str, extra_name: str) -> ModuleType:
    module = optional_import(module_name)
    if module is None:
        raise MissingOptionalDependency(
            f"Missing optional dependency '{module_name}'. Install the '{extra_name}' extras "
            f"or add the package manually."
        )
    return module
