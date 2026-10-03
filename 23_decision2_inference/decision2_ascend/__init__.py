"""Opt-in, process-local extension; never modifies installed framework files."""
import importlib.metadata
import json
import os


def register():
    if os.environ.get("DECISION2_ASCEND_ENABLE") != "1":
        return
    from vllm import __version__
    if __version__ != "0.21.0" or importlib.metadata.version("vllm-ascend") != "0.21.0rc1":
        raise RuntimeError("Decision2 adapter is validated only against vLLM 0.21.0 / Ascend 0.21.0rc1")
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model("Decision2EosForPooling", "decision2_ascend.model:Decision2EosForPooling")
    # The installed HTTP schema permits extras but does not forward them to the
    # pooler. A request-local field avoids mutable IO-processor/global state.
    from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
    if not getattr(PoolingCompletionRequest, "_decision2_patched", False):
        original = PoolingCompletionRequest.to_pooling_params

        def to_pooling_params(self):
            params = original(self)
            metadata = (self.model_extra or {}).get("decision2")
            if metadata is not None:
                from .protocol import validate_metadata
                validate_metadata(self.input, metadata)
                if self.task != "classify" or self.truncate_prompt_tokens is not None:
                    raise ValueError("Decision2 requires classify, without prompt truncation")
                params.extra_kwargs = {"decision2": metadata}
            return params

        PoolingCompletionRequest.to_pooling_params = to_pooling_params
        PoolingCompletionRequest._decision2_patched = True
    print(json.dumps({"event": "decision2_plugin_registered", "pid": os.getpid(),
                      "vllm": __version__, "adapter": __file__}), flush=True)
