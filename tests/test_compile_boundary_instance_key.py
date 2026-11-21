"""Guard the instance key a compile boundary passes across itself.

Inductor caches a compiled artifact on disk with the fake's output shape baked in, and the
key is one of the graph's constants. So a key that two processes share must name instances
whose fakes agree, and a key must resolve to the one instance it was made for.
"""

import dataclasses
import weakref

import pytest
import torch

from tests.fresh_interpreter import probe
from tileops._ops.compile_boundary import get_instance, instance_key, register_instance
from tileops.manifest._values import ADTValue

pytestmark = pytest.mark.smoke


class _Marker:
    """Something to register. `register_instance` only reads the class."""


@dataclasses.dataclass
class _Layout:
    """A dataclass param, as an ADT's `python` class is."""

    align: int
    cache: dict = dataclasses.field(init=False)


def test_a_key_resolves_to_its_own_instance():
    a, b = _Marker(), _Marker()
    ka, kb = (instance_key(register_instance(o), (2,)) for o in (a, b))
    assert ka != kb
    assert get_instance(ka) is a
    assert get_instance(kb) is b
    assert ka.startswith(f"{__name__}._Marker#")
    odd = type("Odd#-Name", (), {})()
    assert get_instance(instance_key(register_instance(odd), (1,))) is odd


def test_a_collected_instances_key_is_never_handed_out_again():
    """An op reaching a used key would inherit that op's compiled shapes."""
    keys = set()
    for _ in range(50):
        op = _Marker()
        keys.add(instance_key(register_instance(op), ()))
        ref = weakref.ref(op)
        del op
        assert ref() is None
    assert len(keys) == 50


def test_a_value_edited_in_place_changes_its_key():
    identity = register_instance(_Marker())
    dims, layout = [1], _Layout(1)
    literal = ADTValue("Layout", {"masked": {"max_m": 4}})
    state = (dims, layout, literal)
    keys = {instance_key(identity, state)}
    dims.append(2)
    keys.add(instance_key(identity, state))
    layout.align = 2
    keys.add(instance_key(identity, state))
    layout.cache = {}
    keys.add(instance_key(identity, state))
    literal.max_m = 8
    keys.add(instance_key(identity, state))
    assert len(keys) == 5
    assert instance_key(identity, (10**5000,)) != instance_key(identity, (10**5000 + 1,))


def test_a_state_the_encoding_does_not_read_keeps_one_key():
    """Too deep, a tensor without strides, or a type it does not read: no error, and the
    same key however the state changes, so a trace never refuses it as stale."""
    identity = register_instance(_Marker())
    deep = {}
    for _ in range(2000):
        deep = {"child": deep}
    states = [(deep,), (torch.eye(2).to_sparse(),), (type("L", (list,), {})(),)]
    keys = {instance_key(identity, state) for state in states}
    deep["other"] = 1
    keys.add(instance_key(identity, states[0]))
    assert len(keys) == 1
    unset, empty = _Layout(1), _Layout(1)
    empty.cache = None
    assert instance_key(identity, (unset,)) != instance_key(identity, (empty,))


def test_cyclic_values_key_by_their_shape():
    """Two cycles that differ in which list a back-reference names key apart."""
    identity = register_instance(_Marker())
    a, b = [1], [2]
    b.append(a)
    a.append(b)
    c, d = [1], [2]
    d.append(d)
    c.append(d)
    assert instance_key(identity, (a,)) != instance_key(identity, (c,))


# Each state's key in a fresh process: stable states, then states with no stable encoding.
_STABLE = (
    "[(2, None, 'x'), (torch.float16, E.A, [1.5, (2,)], {'k': True}), (_Layout(1),),"
    " (1, None, 'x'), (torch.bfloat16, E.A, [1.5, (2,)], {'k': True}), (_Layout(2),)]"
)
# A tensor, an extension type, a function, a container subclass and a scalar subclass.
_LOCAL = (
    "[(torch.zeros(2, 3),), (threading.Lock(),), (lambda: 0,), (type('L', (list,), {})(),),"
    " (type('I', (int,), {})(1),)]"
)


def _keys(member_value: int) -> list[str]:
    """Keys a fresh process gives one `_Marker` holding each state, then a local class's key;
    its enum `E.A` has *member_value*."""
    setup = (
        "import enum, threading, torch\n"
        "from tileops._ops.compile_boundary import instance_key, register_instance\n"
        "from tests.test_compile_boundary_instance_key import _Layout, _Marker\n"
        f"class E(enum.Enum):\n    A = {member_value}\n"
        "def made():\n    class Local: pass\n    return Local()\n"
        "m = register_instance(_Marker())"
    )
    expr = (
        f"' '.join([*(instance_key(m, s) for s in {_STABLE} + {_LOCAL}),"
        " instance_key(register_instance(made()), ())])"
    )
    return probe(setup, expr).split()


def test_two_processes_share_a_key_exactly_when_the_state_agrees():
    """A key names the same fake in every process: an ordinal alone could name instances
    of different params in two runs, and the second would load the first's compiled
    artifact."""
    first, second, other = _keys(1), _keys(1), _keys(2)
    stable = 6
    assert first[:stable] == second[:stable]
    # Each changed state keys apart from its original.
    assert not set(first[:3]) & set(first[3:stable])
    # The local values, and a class defined inside a function: never shared.
    assert not set(first[stable:]) & set(second[stable:])
    # One enum name, another member value, which expressions read.
    assert first[1] != other[1] and first[4] != other[4] and first[0] == other[0]
