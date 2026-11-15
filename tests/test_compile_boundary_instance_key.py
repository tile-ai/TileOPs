"""Guard the instance key a compile boundary passes across itself.

Inductor caches a compiled artifact on disk with the fake's output shape baked in, and the
key is one of the graph's constants. So a key that two processes share must name instances
whose fakes agree, and a key must resolve to the one instance it was made for.
"""

import enum

import pytest

from tests.fresh_interpreter import probe
from tileops._ops.compile_boundary import get_instance, instance_key, register_instance

pytestmark = pytest.mark.smoke


class _Marker:
    """Something to register. `register_instance` only reads the class."""


class _Mode(enum.Enum):
    A = 1


def test_two_instances_of_one_class_get_different_keys():
    a, b = _Marker(), _Marker()
    ka, kb = (instance_key(register_instance(o), (2,)) for o in (a, b))
    assert ka != kb
    assert get_instance(ka) is a
    assert get_instance(kb) is b
    assert ka.startswith(f"{__name__}._Marker#")


def _keys(states: str) -> list[str]:
    """The keys a fresh process gives one new `_Marker` per state in *states*, in order."""
    setup = (
        "import enum, torch\n"
        "from tileops._ops.compile_boundary import instance_key, register_instance\n"
        "from tests.test_compile_boundary_instance_key import _Marker, _Mode"
    )
    expr = f"' '.join(instance_key(register_instance(_Marker()), s) for s in {states})"
    return probe(setup, expr).split()


def test_a_second_process_shares_a_key_exactly_when_the_state_agrees():
    """The failure this guards: `MaxPool1dFwdOp#3` named a stride-2 instance in one run and
    a stride-1 instance in the next, so the second run loaded the first run's artifact and
    asserted on its output shape."""
    states = (
        "[(2, None, 'x'), (torch.float16, _Mode.A, [1.5, (2,)], {'k': True}),"
        " (torch.zeros(2, 3).t(),)]"
    )
    first, second = _keys(states), _keys(states)
    assert first == second
    changed = _keys(
        "[(1, None, 'x'), (torch.bfloat16, _Mode.A, [1.5, (2,)], {'k': True}),"
        " (torch.zeros(2, 3),)]"
    )
    assert not set(first) & set(changed)


def test_a_state_without_a_stable_encoding_keys_apart_in_each_process():
    """An object with no fields to compare, and a class another process may define otherwise,
    never share a key across processes."""
    first, second = _keys("[(object(),), (lambda: 0,)]"), _keys("[(object(),), (lambda: 0,)]")
    assert not set(first) & set(second)
    setup = (
        "from tileops._ops.compile_boundary import instance_key, register_instance\n"
        "def made():\n    class Local: pass\n    return Local()"
    )
    expr = "instance_key(register_instance(made()), ())"
    assert probe(setup, expr) != probe(setup, expr)


class _Size(int):
    """An `int` subclass: its value lives in the builtin base, not in `__dict__`."""


def test_values_that_differ_only_where_the_encoding_cannot_see_key_apart():
    """Unequal `int` subclass values; and, across processes, enum members of one name whose
    values differ, which expressions read."""
    identity = register_instance(_Marker())
    assert instance_key(identity, (_Size(1),)) != instance_key(identity, (_Size(2),))
    setup = (
        "import enum\n"
        "from tileops._ops.compile_boundary import instance_key, register_instance\n"
        "class E(enum.Enum):\n    A = {}"
    )
    expr = "instance_key(register_instance(E), (E.A,))"
    assert probe(setup.format(1), expr) != probe(setup.format(2), expr)


def test_a_class_name_holding_the_separators_resolves_its_instance():
    marker = type("Odd#-Name", (), {})()
    key = instance_key(register_instance(marker), (1,))
    assert get_instance(key) is marker
