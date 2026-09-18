from __future__ import annotations

import pickle

import pytest
import torch
from PIL import Image

from qwen_latent_cot.bagel.rewards import (
    GenEvalRewardClient,
    audit_bagel_flux_vae_contract,
    unpack_bagel_latents,
)


def test_unpack_bagel_latents_preserves_flux_geometry():
    packed = torch.arange(32 * 32 * 64).reshape(1, 32 * 32, 64)
    unpacked = unpack_bagel_latents(packed, (512, 512))
    assert unpacked.shape == (1, 16, 64, 64)
    repacked = unpacked.reshape(1, 16, 32, 2, 32, 2)
    repacked = torch.einsum("nchpwq->nhwpqc", repacked).reshape(1, 32 * 32, 64)
    assert torch.equal(repacked, packed)


class _Response:
    def __init__(self, body):
        self.content = pickle.dumps(body)

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, _url, *, data, timeout):
        assert timeout == 3.0
        self.payload = pickle.loads(data)
        return _Response({"scores": [0.25, 0.75]})


def test_geneval_client_uses_reference_wire_contract():
    client = GenEvalRewardClient(timeout_seconds=3.0)
    client.session = _Session()
    images = [Image.new("RGB", (8, 8), color="red"), Image.new("RGB", (8, 8))]
    metadata = [{"tag": "red"}, {"tag": "black"}]
    scores = client.score(images, metadata, only_strict=True)
    assert torch.equal(scores, torch.tensor([0.25, 0.75]))
    assert set(client.session.payload) == {"images", "meta_datas", "only_strict"}
    assert client.session.payload["meta_datas"] == metadata
    assert client.session.payload["only_strict"] is True
    assert all(isinstance(value, bytes) for value in client.session.payload["images"])


def test_bagel_flux_vae_contract_is_explicit_and_fail_closed():
    class VAE:
        latent_channels = 16
        latent_downsample = 8
        scale_factor = 0.3611
        shift_factor = 0.1159

    assert audit_bagel_flux_vae_contract(VAE())["latent_channels"] == 16
    VAE.scale_factor = 1.0
    with pytest.raises(RuntimeError, match="not compatible"):
        audit_bagel_flux_vae_contract(VAE())
