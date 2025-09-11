import torch
import os

def _resolve_dtype(dtype_spec):
    """
    Map common dtype strings (e.g., 'bfloat16', 'float16', 'float32') to torch dtypes.
    If dtype_spec is already a torch.dtype, return it directly.
    """
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        table = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "f16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "f32": torch.float32,
            "float": torch.float32,
            "half": torch.float16,
        }
        return table.get(key, torch.float32)
    # Fallback
    return torch.float32

def _visible_physical_gpu_ids() -> list[int]:
    """
    Resolve the list of physical GPU IDs from CUDA_VISIBLE_DEVICES.
    If not set, fallback to [0, 1, ..., torch.cuda.device_count()-1].
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in env.split(",") if x.strip() != ""]