from unittest.mock import MagicMock

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.nested_tensor  # noqa: E402
from comfy_extras.nodes_audio import vae_decode_audio  # noqa: E402


def _vae(latent_channels, decoded_shape):
    vae = MagicMock()
    vae.latent_channels = latent_channels
    vae.decode.return_value = torch.zeros(decoded_shape)
    return vae


def test_vae_decode_audio_raises_on_channel_mismatch():
    # MiniMax H3's raw audio latent stream is [B, 32, 2, T]; decoding it with a
    # VAE trained for a different channel count (e.g. an 8-channel audio VAE)
    # must not silently reach a confusing broadcast error deep inside the VAE.
    vae = _vae(latent_channels=8, decoded_shape=(1, 2, 207, 2))
    samples = {"samples": torch.zeros(1, 32, 2, 207)}

    with pytest.raises(ValueError, match="channels"):
        vae_decode_audio(vae, samples)

    vae.decode.assert_not_called()


def test_vae_decode_audio_unwraps_nested_and_decodes_matching_latent():
    video = torch.zeros(1, 24, 2, 8, 8)
    audio = torch.zeros(1, 32, 2, 207)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    vae = _vae(latent_channels=32, decoded_shape=(1, 2, 207, 2))
    vae_decode_audio(vae, samples)

    decoded_arg = vae.decode.call_args[0][0]
    assert decoded_arg is audio
