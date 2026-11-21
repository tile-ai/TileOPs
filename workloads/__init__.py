"""Shared workload definitions for benchmarks and tests.

This package provides WorkloadBase (input generation + workload parameters)
and FixtureBase (reusable pytest parametrize decorators).  Benchmarks and
tests both import from here.  A workload class named for one op also owns
that op's ref_program and its verification() declaration.

workloads/ must not own:
- calculate_flops() / calculate_memory()
- the choice of what a benchmark times against
- scheduling, timing or reporting

Those are decisions and belong to the consumer that makes them.
"""
