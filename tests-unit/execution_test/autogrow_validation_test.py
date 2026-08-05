import asyncio

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import execution
import nodes
from comfy_api.latest import io


class _StubAutogrowNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplateNames(
            io.String.Input("value"),
            names=[f"value_{i}" for i in range(1, 4)],
            min=0,
        )
        return io.Schema(
            node_id="StubAutogrowNode",
            category="_for_testing",
            inputs=[
                io.Autogrow.Input("values", template=template, optional=True),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, values=None):
        return io.NodeOutput("|".join((values or {}).values()))


class _StubSourceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StubSourceNode",
            category="_for_testing",
            inputs=[],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput("hi")


@pytest.fixture(autouse=True)
def register_stub_nodes():
    nodes.NODE_CLASS_MAPPINGS["StubAutogrowNode"] = _StubAutogrowNode
    nodes.NODE_CLASS_MAPPINGS["StubSourceNode"] = _StubSourceNode
    yield
    nodes.NODE_CLASS_MAPPINGS.pop("StubAutogrowNode", None)
    nodes.NODE_CLASS_MAPPINGS.pop("StubSourceNode", None)


def _make_prompt(autogrow_inputs):
    return {
        "1": {"class_type": "StubSourceNode", "inputs": {}},
        "2": {"class_type": "StubAutogrowNode", "inputs": autogrow_inputs},
    }


def test_nested_dict_link_is_rejected():
    """A nested dict matching the Python kwarg shape must not silently pass validation."""
    prompt = _make_prompt({"values": {"value_1": ["1", 0]}})
    valid, errors, _ = asyncio.run(execution.validate_inputs("test-prompt", prompt, "2", {}))
    assert valid is False
    assert any(e["type"] == "bad_autogrow_input_shape" for e in errors)


def test_dotted_key_link_is_accepted():
    """The documented dotted sub-slot key format must keep working."""
    prompt = _make_prompt({"values.value_1": ["1", 0]})
    valid, errors, _ = asyncio.run(execution.validate_inputs("test-prompt", prompt, "2", {}))
    assert valid is True
    assert errors == []


def test_no_autogrow_input_is_accepted():
    prompt = _make_prompt({})
    valid, errors, _ = asyncio.run(execution.validate_inputs("test-prompt", prompt, "2", {}))
    assert valid is True
    assert errors == []
