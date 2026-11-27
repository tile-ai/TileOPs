"""Workload definitions benchmarks and tests share: inputs, references and verification.

workloads/ must not own:
- calculate_flops() / calculate_memory()
- the choice of what a benchmark times against
- scheduling, timing or reporting

Those are decisions and belong to the consumer that makes them.
"""
