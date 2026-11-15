"""Guard the instance key a compile boundary passes across itself.

Inductor caches a compiled artifact on disk with the fake's output shape baked in, and the
key is one of the graph's constants. So a key that two processes share must name instances
whose fakes agree, and a key must resolve to the one instance it was made for.
"""

import pytest

from tests.fresh_interpreter import probe
from tileops._ops.compile_boundary import get_instance, instance_key, register_instance

pytestmark = pytest.mark.smoke


class _Marker:
    """Something to register. `register_instance` only reads the class."""


class _Size(int):
    """An `int` subclass: its value lives in the builtin base, not in `__dict__`."""


class _Dims(list):
    """A `list` subclass a param may hold and edit in place."""


def test_a_key_resolves_to_its_own_instance():
    a, b = _Marker(), _Marker()
    ka, kb = (instance_key(register_instance(o), (2,)) for o in (a, b))
    assert ka != kb
    assert get_instance(ka) is a
    assert get_instance(kb) is b
    assert ka.startswith(f"{__name__}._Marker#")
    odd = type("Odd#-Name", (), {})()
    assert get_instance(instance_key(register_instance(odd), (1,))) is odd


def test_unequal_values_the_encoding_cannot_see_key_apart_in_one_process():
    identity = register_instance(_Marker())
    assert instance_key(identity, (_Size(1),)) != instance_key(identity, (_Size(2),))
    dims = _Dims([1])
    before = instance_key(identity, (dims,))
    dims.append(2)
    assert instance_key(identity, (dims,)) != before


# Each state's key in a fresh process: stable states, then states with no stable encoding.
_STABLE = (
    "[(2, None, 'x'), (torch.float16, E.A, [1.5, (2,)], {'k': True}), (torch.zeros(3, 2).t(),),"
    " (1, None, 'x'), (torch.bfloat16, E.A, [1.5, (2,)], {'k': True}), (torch.zeros(2, 3),)]"
)
_LOCAL = "[(object(),), (lambda: 0,)]"


def _keys(member_value: int) -> list[str]:
    """Keys a fresh process gives one `_Marker` holding each state, then a local class's key;
    its enum `E.A` has *member_value*."""
    setup = (
        "import enum, torch\n"
        "from tileops._ops.compile_boundary import instance_key, register_instance\n"
        "from tests.test_compile_boundary_instance_key import _Marker\n"
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
    """The failure this guards: `MaxPool1dFwdOp#3` named a stride-2 instance in one run and
    a stride-1 instance in the next, so the second run loaded the first run's artifact and
    asserted on its output shape."""
    first, second, other = _keys(1), _keys(1), _keys(2)
    stable = 6
    assert first[:stable] == second[:stable]
    # Each changed state keys apart from its original.
    assert not set(first[:3]) & set(first[3:stable])
    # No stable encoding, or a class defined inside a function: never shared.
    assert not set(first[stable:]) & set(second[stable:])
    # One enum name, another member value, which expressions read.
    assert first[1] != other[1] and first[4] != other[4] and first[0] == other[0]
