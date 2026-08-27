# Import-guarded so node functions remain importable without the Kedro runtime
# (e.g. in lightweight CI). When Kedro is installed, create_pipeline is exposed
# normally for the pipeline registry.
try:
    from .pipeline import create_pipeline  # noqa: F401
    __all__ = ["create_pipeline"]
except ModuleNotFoundError:  # pragma: no cover
    __all__ = []
