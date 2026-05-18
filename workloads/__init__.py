"""Shared workload definitions for benchmarks and tests.

This package provides WorkloadBase (input generation + workload parameters)
and FixtureBase (reusable pytest parametrize decorators).  Benchmarks and
tests both import from here.

workloads/ must not own:
- ref_program() or other reference implementations
- check() or assertion/tolerance logic
- any correctness-only semantics

Those belong in tests/.
"""
