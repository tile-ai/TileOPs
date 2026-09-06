<!--
Keep it to what changed and how to verify it. One line per bullet.
Delete any section that does not apply — never leave an empty header.
Do not recount the development process or restate a linked issue's motivation.
Conventions: CONTRIBUTING.md
-->

## Summary

-
-

## Test plan

- [ ] `pre-commit run --all-files`
- [ ] `pytest` on the affected files, on a real GPU

## Test node delta

<!-- Required when the PR touches tests/. Paste the output of
     `python scripts/test_node_delta.py --base upstream/main`, and give one line
     per new case naming the code path, dtype, feature, or regression it guards. -->

## Benchmark

<!-- Required for a kernel or op change. One section per op, one row per
     (shape, dtype). Keep the shape as a tuple: (4096, 4096), not 16M.
     Speedup is baseline_ms / tileops_ms. Show one throughput column —
     TFLOPS for compute-bound ops, BW (TB/s) for memory-bound ones. -->

**Environment**: \{GPU}, CUDA \{ver}, PyTorch \{ver}, TileLang \{ver}

### \{OpName}

| Shape | dtype | TileOPs (ms) | \{baseline} (ms) | Speedup | BW (TB/s) |
| ----- | ----- | ------------ | ---------------- | ------- | --------- |
|       |       |              |                  |         |           |

**Takeaways:**

## Regression

<!-- Recommended for a bugfix or refactor: what would have caught this. -->
