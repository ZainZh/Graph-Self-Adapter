"""PROVE-Embodied runtime contracts.

The public namespace is deliberately small. Foundation capabilities are
looked up by stable prove.*@version references; search candidates compose
those capabilities through ObserverSpec rather than importing backend code.
"""

from .observation import (
    build_foundation_registry,
    build_foundation_tool_index,
    load_foundation_catalog,
)

__all__ = [
    "build_foundation_registry",
    "build_foundation_tool_index",
    "load_foundation_catalog",
    "tools",
]
