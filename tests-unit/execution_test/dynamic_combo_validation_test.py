"""Regression test for #16236: a stale/renamed DynamicCombo option key (e.g. a
saved workflow predating the option rename in BlockSparseAttention's `selection`
input) must be rejected during prompt validation instead of silently reaching
execute(), where it crashes with a raw KeyError/AttributeError deep inside the
node body."""
import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution  # noqa: E402
import nodes  # noqa: E402
from comfy_extras.nodes_sparse_attention import BlockSparseAttention  # noqa: E402

pytestmark = pytest.mark.asyncio


class _FakeModelSource:
    """Minimal upstream node satisfying BlockSparseAttention's MODEL input link."""
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def go(self):
        return (None,)


def _prompt_with_selection(selection):
    return {
        "1": {"class_type": "_FakeModelSource", "inputs": {}},
        "2": {
            "class_type": "BlockSparseAttention",
            "inputs": {
                "model": ["1", 0],
                "selection": selection,
                "start_percent": 0.2,
                "end_percent": 1.0,
                "dense_blocks": "",
                "min_tokens": 12288,
                "extra_tokens": 256,
                "sink_conditioning": "exact_kv_and_rows",
                "verbose": False,
            },
        },
    }


@pytest.fixture(autouse=True)
def _register_test_nodes():
    prior = nodes.NODE_CLASS_MAPPINGS.get("BlockSparseAttention")
    nodes.NODE_CLASS_MAPPINGS["BlockSparseAttention"] = BlockSparseAttention
    nodes.NODE_CLASS_MAPPINGS["_FakeModelSource"] = _FakeModelSource
    yield
    if prior is None:
        nodes.NODE_CLASS_MAPPINGS.pop("BlockSparseAttention", None)
    else:
        nodes.NODE_CLASS_MAPPINGS["BlockSparseAttention"] = prior
    nodes.NODE_CLASS_MAPPINGS.pop("_FakeModelSource", None)


async def test_stale_dynamic_combo_selection_rejected_before_execute():
    prompt = _prompt_with_selection("no-longer-exists")

    valid, errors, node_id = await execution.validate_inputs("test-prompt", prompt, "2", {})

    assert valid is False
    assert node_id == "2"
    assert any(
        e["type"] == "value_not_in_list" and e["extra_info"]["input_name"] == "selection"
        for e in errors
    )


async def test_matching_dynamic_combo_selection_still_validates():
    prompt = _prompt_with_selection("sol-attn")
    prompt["2"]["inputs"]["selection.tau"] = 1.3

    valid, errors, _ = await execution.validate_inputs("test-prompt", prompt, "2", {})

    assert valid is True, errors
