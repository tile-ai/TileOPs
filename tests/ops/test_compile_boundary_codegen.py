"""The operator schema this package derives from a manifest entry.

Five forms carry every op on the boundary: an optional input, several outputs, an
argument the operator writes, a buffer appended after the declared inputs, and a
declared input the operator writes into. They are checked here against the entry they
come from, so a schema that stops matching its declaration fails without a GPU.
"""

import pytest

from tileops.ops._compile_boundary_codegen import (
    OperatorSpec,
    operator_name,
    operator_schema,
    written_arguments,
)


def _entry(inputs: dict, outputs: tuple, family: str = "demo") -> dict:
    return {
        "family": family,
        "signature": {"inputs": inputs, "outputs": dict.fromkeys(outputs, {})},
    }


@pytest.mark.smoke
def test_a_required_input_and_one_output() -> None:
    entry = _entry({"x": {}, "weight": {}}, ("output",))
    schema = operator_schema(entry, OperatorSpec(), ())

    assert schema == "(Tensor x, Tensor weight, str instance_key) -> Tensor"


@pytest.mark.smoke
def test_an_optional_input_is_marked_optional() -> None:
    entry = _entry({"x": {}, "bias": {"optional": True}}, ("output",))

    assert operator_schema(entry, OperatorSpec(), ()) == (
        "(Tensor x, Tensor? bias, str instance_key) -> Tensor"
    )


@pytest.mark.smoke
def test_several_outputs_come_back_as_a_tuple() -> None:
    entry = _entry({"x": {}}, ("dq", "dk", "dv"))

    assert operator_schema(entry, OperatorSpec(), ()) == (
        "(Tensor x, str instance_key) -> (Tensor, Tensor, Tensor)"
    )


@pytest.mark.smoke
def test_a_written_input_carries_the_alias_and_returns_nothing() -> None:
    entry = _entry({"input": {"mutated": True}}, ("output",))
    spec = OperatorSpec.inplace("input")

    assert operator_schema(entry, spec, ("input",)) == (
        "(Tensor(a!) input, str instance_key) -> ()"
    )


@pytest.mark.smoke
def test_a_caller_supplied_buffer_is_appended_after_the_declared_inputs() -> None:
    entry = _entry({"a": {}, "b": {}}, ("output",))
    spec = OperatorSpec.writes_out("out")

    assert operator_schema(entry, spec, ("out",)) == (
        "(Tensor a, Tensor b, Tensor(a!) out, str instance_key) -> ()"
    )


@pytest.mark.smoke
def test_a_written_buffer_that_is_a_declared_input_is_not_appended() -> None:
    entry = _entry(
        {"output": {"mutated": True}, "x": {}, "workspace": {"mutated": True}},
        ("output",),
    )
    spec = OperatorSpec.writes_out("output")
    mutates = written_arguments(entry, spec, (spec,))

    assert mutates == ("output", "workspace")
    assert operator_schema(entry, spec, mutates) == (
        "(Tensor(a!) output, Tensor x, Tensor(b!) workspace, str instance_key) -> ()"
    )


class TestWrittenArguments:
    """Which arguments an operator writes, read off the manifest and the spec set."""

    @pytest.mark.smoke
    def test_one_operator_writes_what_the_manifest_marks(self) -> None:
        entry = _entry({"x": {}, "running_mean": {"mutated": True}}, ("output",))
        default = OperatorSpec()

        assert written_arguments(entry, default, (default,)) == ("running_mean",)

    @pytest.mark.smoke
    def test_a_writing_sibling_takes_the_writes_off_the_default(self) -> None:
        entry = _entry({"input": {"mutated": True}}, ("output",))
        default, inplace = OperatorSpec(), OperatorSpec.inplace("input")
        specs = (default, inplace)

        assert written_arguments(entry, default, specs) == ()
        assert written_arguments(entry, inplace, specs) == ("input",)

    @pytest.mark.smoke
    def test_a_writing_operator_also_writes_the_marked_inputs(self) -> None:
        entry = _entry({"a": {}, "scratch": {"mutated": True}}, ("output",))
        spec = OperatorSpec.writes_out("out")

        assert written_arguments(entry, spec, (spec,)) == ("scratch", "out")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "family, class_name, expected",
    [
        ("normalization", "RMSNormFwdOp", "normalization_rms_norm_fwd"),
        ("convolution", "Conv2dFwdOp", "convolution_conv2d_fwd"),
        ("moe", "MoePrePermuteFwdOp", "moe_pre_permute_fwd"),
    ],
)
def test_the_operator_names_its_family_once(family: str, class_name: str, expected: str) -> None:
    assert operator_name(family, class_name) == expected
