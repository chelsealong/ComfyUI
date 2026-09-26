import torch

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_patcher
from comfy import ops


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ops.manual_cast.Linear(4, 4, device="cpu", dtype=torch.float32)


def test_partially_unload_does_not_duplicate_lowvram_patches():
    model = TinyModel()
    model.layer1.weight_function = []
    model.layer1.bias_function = []

    patcher = comfy.model_patcher.ModelPatcher(model, load_device="cpu", offload_device="cpu")
    diff = torch.ones(4, 4)
    patcher.patches["layer1.weight"] = [(1.0, (diff,), 1.0, None, None)]

    # First call: module is "fully loaded" and gets partially unloaded, picking up a
    # LowVramPatch hook for the LoRA-style diff patch above.
    model.layer1.comfy_patched_weights = True
    patcher.partially_unload("cpu", memory_to_free=10 ** 9)
    assert len(model.layer1.weight_function) == 1

    # A later full load of the same module flips comfy_patched_weights back to True
    # without clearing the stale weight_function list. A second partially_unload of
    # the same module must not stack another duplicate LowVramPatch on top of it.
    model.layer1.comfy_patched_weights = True
    patcher.partially_unload("cpu", memory_to_free=10 ** 9)
    assert len(model.layer1.weight_function) == 1
