"""Release parsed GE graph definitions after their executors are loaded."""

from __future__ import annotations

from collections.abc import Iterable
import types
from typing import Any


_COMPACTABLE_GRAPH_FIELDS = (
    "_model",
    "_proto",
    "_python_code",
    "_generator_rng_state",
    "_indexed_inputs",
    "_named_inputs_info",
    "_used_process_group",
    "_dont_prune_me_ops",
    "op_name_idx_dict",
    "op_name_list",
)


def _walk_compiled_functions(root: Any) -> Iterable[types.FunctionType]:
    """Walk only callable wrappers and closures, never model or tensor trees."""
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, types.MethodType):
            pending.append(value.__func__)
            continue
        if isinstance(value, types.FunctionType):
            yield value
            if value.__closure__:
                for cell in value.__closure__:
                    try:
                        member = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(
                        member,
                        (types.FunctionType, types.MethodType),
                    ):
                        pending.append(member)
            for name, member in value.__globals__.items():
                if name.startswith("__compiled_fn") and isinstance(
                    member,
                    (types.FunctionType, types.MethodType),
                ):
                    pending.append(member)
            continue
        compiled_model = getattr(value, "_compiled_model", None)
        if compiled_model is not None:
            pending.append(compiled_model)


def compact_loaded_ge_graphs(callables: Iterable[Any]) -> dict[str, Any]:
    """Drop graph-build data after every generated kernel completed first call.

    TorchAir's generated kernel enters graph load and compile only while its
    module-global ``_is_first_run`` flag is true. After a successful first call,
    later calls use only ``ge_graph._executor.run``. The Python protobuf model,
    serialized graph bytes, and graph-construction dictionaries are redundant.
    """
    kernels: dict[int, types.FunctionType] = {}
    for callable_object in callables:
        for function in _walk_compiled_functions(callable_object):
            globals_dict = function.__globals__
            if "ge_graph" not in globals_dict or "_is_first_run" not in globals_dict:
                continue
            kernels[id(globals_dict)] = function

    reports = []
    for function in kernels.values():
        globals_dict = function.__globals__
        if bool(globals_dict.get("_is_first_run", True)):
            raise RuntimeError(
                "cannot compact a TorchAir GE graph before its first call"
            )
        ge_graph = globals_dict["ge_graph"]
        executor = getattr(ge_graph, "_executor", None)
        if executor is None:
            raise RuntimeError("loaded GE graph has no executor")
        serialized_graph = globals_dict.get("serialized_graph")
        serialized_bytes = (
            len(serialized_graph)
            if isinstance(serialized_graph, (bytes, bytearray))
            else 0
        )
        model = getattr(ge_graph, "_model", None)
        try:
            model_bytes = int(model.ByteSize()) if model is not None else 0
        except Exception:
            model_bytes = 0

        cleared_fields = []
        for field in _COMPACTABLE_GRAPH_FIELDS:
            if not hasattr(ge_graph, field):
                continue
            current = getattr(ge_graph, field)
            if isinstance(current, dict):
                current.clear()
            elif isinstance(current, list):
                current.clear()
            elif field == "_python_code":
                setattr(ge_graph, field, "")
            else:
                setattr(ge_graph, field, None)
            cleared_fields.append(field)
        globals_dict["serialized_graph"] = None
        globals_dict["local_compile_options"] = None
        reports.append(
            {
                "kernel": function.__qualname__,
                "serialized_graph_bytes": serialized_bytes,
                "parsed_model_bytes": model_bytes,
                "cleared_fields": cleared_fields,
                "executor_type": type(executor).__name__,
            }
        )

    return {
        "graph_count": len(reports),
        "serialized_graph_bytes": sum(
            int(item["serialized_graph_bytes"]) for item in reports
        ),
        "parsed_model_bytes": sum(
            int(item["parsed_model_bytes"]) for item in reports
        ),
        "graphs": reports,
    }


def release_loaded_ge_executors(callables: Iterable[Any]) -> dict[str, Any]:
    """Drop cache-loaded GE executors while retaining lazy cache loaders."""
    released = []
    seen: set[int] = set()
    for callable_object in callables:
        identity = id(callable_object)
        if identity in seen:
            continue
        seen.add(identity)
        if not hasattr(callable_object, "_compiled_model"):
            raise TypeError(
                "executor release requires a TorchAir LazyCompiledModel, got "
                f"{type(callable_object)!r}"
            )
        compiled_model = getattr(callable_object, "_compiled_model")
        if compiled_model is None:
            raise RuntimeError("TorchAir executor is not loaded")
        setattr(callable_object, "_compiled_model", None)
        released.append(
            {
                "callable_type": type(callable_object).__name__,
                "compiled_model_type": type(compiled_model).__name__,
            }
        )
    return {"executor_count": len(released), "executors": released}
