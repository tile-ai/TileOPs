"""Shared workload definitions for benchmarks and tests.

This package provides WorkloadBase (input generation + workload parameters)
and FixtureBase (reusable pytest parametrize decorators).  Benchmarks and
tests both import from here.  A workload class named for one op also owns
that op's ref_program.

workloads/ must not own:
- check() or assertion/tolerance logic
- calculate_flops() / calculate_memory()
- the choice of what a benchmark times against

Those are decisions and belong to the consumer that makes them.
"""
