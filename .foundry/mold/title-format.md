# Issue Title Format

**Format:** `[TYPE][COMPONENT] short description in lowercase`

- **TYPE**: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META | BENCHMARK — canonical list in `.claude/conventions/types.sh`
- **COMPONENT**: mandatory — kernel name or subsystem (GEMV, GEMM, FLASH_ATTN, CI, TOOLING, …)
- Max 80 characters total, description in lowercase
- Do NOT use Conventional Commits style (`feat(scope): …`)

Examples: `[FEAT][GEMV] add batched forward pass` · `[BUG][FLASH_ATTN] fix bf16 overflow in softmax` · `[PERF][GEMM] optimize shared memory layout for H100`
