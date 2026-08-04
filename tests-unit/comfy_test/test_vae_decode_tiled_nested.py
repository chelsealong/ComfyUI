"""VAEDecodeTiled must unbind NestedTensor latents (e.g. MiniMax H3 AV pairs)
before handing them to VAE.decode_tiled, matching VAEDecode's behavior."""

from __future__ import annotations

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.nested_tensor  # noqa: E402
import nodes as nodes_mod  # noqa: E402


class _StubVAE:
    def __init__(self):
        self.received = None

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 8

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        self.received = samples
        return torch.zeros(1, 4, 4, 3)


def test_decode_tiled_unbinds_nested_latent_before_calling_vae():
    video = torch.full((1, 24, 2, 4, 4), 1.0)
    audio = torch.full((1, 32, 2, 10), 2.0)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    vae = _StubVAE()
    nodes_mod.VAEDecodeTiled().decode(vae, samples, tile_size=512, overlap=64)

    assert torch.is_tensor(vae.received)
    assert torch.equal(vae.received, video)
