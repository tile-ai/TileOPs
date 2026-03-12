# TileOPs Collaboration Guide

This document outlines the standard workflow for contributing to TileOPs. We follow a strict **"2+1 Review"** policy to ensure code quality and knowledge sharing.

## 1. The Development Lifecycle

We follow a standard **Issue -> Fork -> PR** workflow. **Do not** create branches directly on the main repository.

### Step 1: Issue (Pick or Create a Task)

- **New Operators**: If you represent a new operator, **Create a New Issue** using the "New Operator Request" template.
- **Existing Tasks**: Browse [Issues](https://github.com/tile-ai/TileOPs/issues) and comment to assign it to yourself.
- **Decompose**: For "New Operator" epics, create **Sub-task Issues** (e.g., "Implement Kernel") using the template.

### Step 2: Fork & Branch (Start Work)

- **Fork**: Fork the repository to your GitHub account.
- **Clone**: Clone your fork locally (`git clone ...`).
- **Setup**: Run `make install` to install dev dependencies and pre-commit hooks in one step.
- **Branch**: Create a new branch on your fork.
  - **Base**: Sync with upstream `main` before branching.
  - **Name**: `type/scope/description` (canonical prefixes in `.claude/conventions/types.sh`), for example:
    - `feat/flash-attn/fwd-kernel`
    - `fix/mla/parsing-error`
    - `doc/readme/add-examples`
    - `test/gemv/add-edge-cases`
    - `perf/mha/improve-bandwidth`
    - `bench/gemm/add-triton-baseline`

### Step 3: Commit (Save Work)

We follow the **TileLang Commit Convention**: `[Type] Description` or `[Type][Scope] Description`.

**Common Types** (canonical list in `.claude/conventions/types.sh`):

- `[Feat]`: New features or operators.
- `[BugFix]`: Bug fixes.
- `[Fix]`: Non-bug corrections.
- `[Refactor]`: Code restructuring without behavior change.
- `[Enhancement]`: Improvements to existing features.
- `[Doc]`: Documentation updates.
- `[Chore]`: Build system or workflow changes.
- `[Bench]`: Benchmark updates.
- `[CI]`: CI/CD changes.
- `[Test]`: Test changes.
- `[Perf]`: Performance improvements.

**Examples**:

- `[Feat] Add multi-head attention forward op`
- `[BugFix] Fix index out of bounds in reduction kernel`
- `[Refactor] Reformat code to adhere to Google Style`
- `[Enhancement][MHA] Improve multi-head attention forward op performance on Hopper`
- `[Doc] Update README.md`
- `[Chore][CI] Update CUDA version to 12.9`
- `[Bench][MHA] Add Triton baseline for multi-head attention forward op`

> **Tip**: Run `pre-commit run --all-files` and fix any issues before pushing!
>
> **Commit scope rule**: Keep commit messages concise. Do not put long verification sections,
> benchmark tables, or command transcripts into commit bodies; those belong in the PR description
> and tracking issue.

### Step 4: Pull Request (Submit Code)

- **Title**: Matches your commit, e.g., `[Feat] Add multi-head attention forward op`.
- **Template**: Fill out the PR template checklist fully.
- **Description**: Provide a detailed description of the changes, including any relevant context or background information. You can leverage `gemini-code-assist`'s summary for the PR description.
- **Benchmark Reporting**: For new ops or semantic/performance-sensitive changes, report real measured data from a host-visible CUDA machine. Include representative small, medium, and large shapes.
- **Verification Order**: Run the targeted correctness suite on the same machine before publishing benchmark numbers.
- **CI**: Ensure all GitHub Actions (Lint/Test/Build) pass.

## 2. The "2+1" Review Policy

We use a 2+1 review process to exercise development procedures while maintaining code quality.

### Peer Review (The "2")

- **Goal**: Sanity check, code style, logic verification, and knowledge sharing.
- **Action**: Invite **2 other developers** to review your PR.
- **Requirement**: You must get **2 Approvals** from peers before moving to Phase 2.
  - *Peers should verify: Procedure (Issue-Fork-PR), Logic correctness, Test coverage, CI passing, Formatting, Naming conventions.*

### Mentor Review (The "1")

- **Goal**: Architecture validation, security check, and final gatekeeping.
- **Action**: Once you have 2 peer approvals, request review from the **Mentor Team** (`@tile-ai/tileops-review`).
- **Requirement**: You must get **1 Approval** from the mentor team.
  - *Mentor verification: Architectural fit, Performance implications, Breaking changes.*

### Merge

Only after **2 Peer Approvals + 1 Mentor Approval + CI Passing** can the code be merged.

## 3. Checklist for Reviewers

### For Developers (Peer Review)

**Process & Style**

- [ ] **Workflow**: Is the PR linked to an Issue? Is it from a Fork?
- [ ] **CI/CD**: Did all automated checks (Lint, Test, Build) pass?
- [ ] **Formatting**: Does it strictly follow Google Python Style (imports, naming)?
- [ ] **Docstrings**: Are Google-style docstrings present for all public elements?
- [ ] **Docs**: Is the documentation clear, complete, and free of typos?

**Correctness & Testing**

- [ ] **Logic**: Is the algorithm correct? Are there any obvious bugs?
- [ ] **Unit Tests**: Are there tests in `tests/` matching the code? Do they pass?
- [ ] **Shared Harness**: If the PR changes shared test infrastructure such as `tests/test_base.py` or common fixtures, does it preserve existing default semantics or update all affected tests in the same PR?
- [ ] **Edge Cases**: Are empty inputs or boundary shapes handled?
- [ ] **Error Handling**: Are inputs validated with informative error messages?
- [ ] **Smoke Scope**: If shared test infrastructure changed, was a broader real-machine `pytest -m smoke` run completed in addition to targeted suites?

**Benchmark**

- [ ] **Benchmark**: Are benchmark scripts provided in `benchmarks/`?
- [ ] **Results**: Are latency / TFLOPS / bandwidth numbers included in the issue and PR description?
- [ ] **Runtime**: Were GPU-dependent tests and benchmarks run on a real CUDA-visible machine rather than a sandbox-only environment?
- [ ] **Coverage**: Does the benchmark table include representative small, medium, and large shapes?
- [ ] **Honesty**: Does the benchmark table report representative regressions as-is instead of omitting them?

### For Mentors (Mentor Review)

**Architecture & Design**

- [ ] **2-Layer Compliance**: Does it cleanly separate `Kernel` (L1) -> `Op` (L2)?
- [ ] **API Design**: Is the L2 Op interface Pythonic and standard?
- [ ] **Compatibility**: Is the L2 Op compatible with `torch.compile` and CUDA Graphs?

**Maintenance**

- [ ] **Breaking Changes**: Does this break existing APIs? (If so, is it necessary/documented?)
- [ ] **Docs**: Is the documentation clear, complete, and free of typos?
