For `[Bench]` PRs. A benchmark measures. It is not where an op is shown to be correct, and it is not a second manifest.

Load `.claude/domain-rules/benchmark.md` first — those are the rules. This is what to look at in a diff.

#### Checklist

- [ ] **Something other than TileOps is timed.** Timing TileOps against TileOps decides nothing.
- [ ] **Correctness still lives in `tests/`.** A benchmark asserts only what it needs to trust its own numbers — that an implementation it is about to time matches the reference, or that a comparison which decides something came out the way the code assumes. It never becomes the place an op's behaviour is established.
- [ ] **A library a row selects fails that row when absent.** Reporting another implementation under its tag is worse than reporting nothing.
- [ ] **Rows name their op.** The benchmark takes its op at construction and publishes every row under it. What distinguishes one case from another belongs to the case, never to the row's name.
- [ ] **Shapes come from somewhere.** A shape constant names the model or scenario it represents; a round number on its own does not.
