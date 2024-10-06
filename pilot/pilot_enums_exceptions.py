from enum import Enum
ExecutionEngine = Enum('ExecutionEngine', ["RAY", "DASK"])


class ExecutionEngine(Enum):
    DASK = "dask"
    RAY = "ray"

class PilotAPIException(Exception):
    pass    