"""Guard the instance key a compile boundary passes across itself.

Inductor caches a compiled artifact on disk with the fake's output shape baked in, and the
key is one of the graph's constants. So a key that named one instance must never come back
naming another — not later in the process, and not in a later process, which is where the
counter alone used to repeat.
"""

import subprocess
import sys

import pytest

from tileops.ops.compile_boundary import get_instance, register_instance


class _Marker:
    """Something to register. `register_instance` only reads the class name."""


@pytest.mark.smoke
def test_two_instances_of_one_class_get_different_keys():
    a, b = _Marker(), _Marker()
    ka, kb = register_instance(a), register_instance(b)
    assert ka != kb
    assert get_instance(ka) is a
    assert get_instance(kb) is b
    assert ka.startswith("_Marker#")


@pytest.mark.smoke
def test_a_second_process_reuses_no_key():
    """The failure this guards: `MaxPool1dFwdOp#3` named a stride-2 instance in one run and
    a stride-1 instance in the next, so the second run loaded the first run's artifact and
    asserted on its output shape."""
    code = (
        "from tileops.ops.compile_boundary import register_instance\n"
        "class M: pass\n"
        "print(' '.join(register_instance(M()) for _ in range(4)))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.split()
        for _ in range(2)
    ]
    assert runs[0] != runs[1]
    assert not set(runs[0]) & set(runs[1])
