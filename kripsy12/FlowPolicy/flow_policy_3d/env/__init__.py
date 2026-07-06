__all__ = ["AdroitEnv", "MetaWorldEnv"]

import importlib


def __getattr__(name: str):
    if name == "AdroitEnv":
        from .adroit import AdroitEnv

        return AdroitEnv
    if name == "MetaWorldEnv":
        from .metaworld import MetaWorldEnv

        return MetaWorldEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
