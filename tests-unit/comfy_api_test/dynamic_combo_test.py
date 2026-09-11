from comfy_api.latest import io
from comfy_api.latest._io import build_nested_inputs, get_finalized_class_inputs


class _DynamicComboNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_DynamicComboNode",
            inputs=[
                io.DynamicCombo.Input("selection", options=[
                    io.DynamicCombo.Option("sol-attn", [io.Float.Input("tau", default=1.3)]),
                    io.DynamicCombo.Option("sla", [io.Float.Input("keep_percent", default=10.0)]),
                ]),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, selection):
        return io.NodeOutput()


def test_dynamic_combo_resolves_matching_option():
    class_inputs = _DynamicComboNode.INPUT_TYPES()
    live_inputs = {"selection": "sol-attn", "selection.tau": 1.3}

    resolved, _, v3_data = get_finalized_class_inputs(class_inputs, live_inputs)
    assert "selection" in resolved["required"]

    nested = build_nested_inputs(dict(live_inputs), v3_data)
    assert nested["selection"] == {"selection": "sol-attn", "tau": 1.3}


def test_dynamic_combo_keeps_selection_input_for_unknown_option_key():
    """A stale option key (e.g. a saved workflow predating an option rename)
    must not make the whole 'selection' input vanish from the resolved
    schema -- otherwise execute() is called without it and crashes with a
    confusing 'missing required positional argument' TypeError instead of
    the node getting a chance to handle the unrecognized value itself."""
    class_inputs = _DynamicComboNode.INPUT_TYPES()
    live_inputs = {"selection": "no-longer-exists"}

    resolved, _, v3_data = get_finalized_class_inputs(class_inputs, live_inputs)
    assert "selection" in resolved["required"]

    nested = build_nested_inputs(dict(live_inputs), v3_data)
    assert "selection" in nested
    assert nested["selection"]["selection"] == "no-longer-exists"
