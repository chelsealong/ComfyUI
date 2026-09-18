import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
import comfy.model_patcher
import comfy.ops


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = comfy.ops.disable_weight_init.Linear(4, 4, bias=False)


def test_patch_weight_to_device_force_cast_uses_archived_model_dtype():
    # Mirrors how comfy.sd.VAE loads weights: the module is cast to the
    # target dtype first, then a state-dict load can leave a parameter in a
    # different dtype (e.g. a lazy/zero-copy load of a mixed-precision
    # checkpoint). Dynamic VRAM staging later force-casts weights to device
    # via patch_weight_to_device(force_cast=True) and must still restore the
    # intended dtype, or a Linear ends up with a weight dtype that doesn't
    # match its input.
    model = _Tiny()
    model.to(torch.float16)
    comfy.model_management.archive_model_dtypes(model)
    assert model.to_q.weight_comfy_model_dtype == torch.float16

    model.to_q.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16), requires_grad=False)
    assert model.to_q.weight.dtype == torch.bfloat16

    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    patcher.patch_weight_to_device("to_q.weight", device_to=torch.device("cpu"), force_cast=True)

    assert model.to_q.weight.dtype == torch.float16

    x = torch.randn(1, 4, dtype=torch.float16)
    out = model.to_q(x)
    assert out.dtype == torch.float16
