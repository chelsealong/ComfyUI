"""Regression test for #15606: a force-full-load model (e.g. the LTX-2.5 VAE
diffusion decoder, which sets ``disable_offload=True``) needs its memory
available immediately, so it must not be shielded from evicting another
resident dynamic model under the "dynamic models page on demand" rule that
only makes sense when the incoming load is itself dynamic/on-demand."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as mm  # noqa: E402


class _StubModel:
    """Minimal ModelPatcher stand-in exposing only what load_models_gpu touches."""

    def __init__(self, size, dynamic, load_device):
        self._size = size
        self._dynamic = dynamic
        self._loaded = 0
        self.load_device = load_device
        self.parent = None

    def is_dynamic(self):
        return self._dynamic

    def model_patches_models(self):
        return []

    def model_size(self):
        return self._size

    def loaded_size(self):
        return self._loaded

    def loaded_ram_size(self):
        return 0

    def current_loaded_device(self):
        return None

    def is_clone(self, other):
        return False

    def model_patches_to(self, *_a, **_k):
        pass

    def model_dtype(self):
        return torch.float32

    def partially_load(self, device, extra_memory, force_patch_weights=False):
        self._loaded = self._size
        return self._size

    @property
    def model(self):
        return self


def test_force_full_load_frees_memory_from_resident_dynamic_models(monkeypatch):
    device = torch.device("cuda:0")

    dit = _StubModel(size=20 * 1024 ** 3, dynamic=True, load_device=device)
    dit_loaded = mm.LoadedModel(dit)
    dit_loaded.model_load()

    monkeypatch.setattr(mm, "current_loaded_models", [dit_loaded])
    monkeypatch.setattr(mm, "get_free_memory", lambda *a, **k: 1024 ** 3)

    free_memory_mock = MagicMock(return_value=[])
    monkeypatch.setattr(mm, "free_memory", free_memory_mock)

    vae = _StubModel(size=4 * 1024 ** 3, dynamic=True, load_device=device)
    mm.load_models_gpu([vae], memory_required=1024 ** 3, force_full_load=True)

    assert free_memory_mock.call_args_list, "free_memory was never called"
    first_call_kwargs = free_memory_mock.call_args_list[0].kwargs
    assert first_call_kwargs["for_dynamic"] is False, (
        "force_full_load=True must not let the resident DiT be treated as "
        "'freeable on demand' -- it needs to actually be evicted so the "
        "force-loaded model has room, otherwise both models stay fully "
        "resident and the allocator can crash instead of raising a "
        "catchable OOM (issue #15606)."
    )
