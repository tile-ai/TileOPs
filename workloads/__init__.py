"""Shared workload definitions for benchmarks and tests.

This package provides WorkloadBase (input generation + reference programs)
and FixtureBase (reusable pytest parametrize decorators).  Benchmarks and
tests both import from here; correctness-only logic (assertions, tolerances)
stays in tests/.
"""
