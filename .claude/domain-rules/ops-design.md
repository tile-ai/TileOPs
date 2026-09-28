→ [layer-boundaries.md §Implementation](../../docs/design/layer-boundaries.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

- Class names: PascalCase `{Name}{Direction}Op` (Op layer) or `{Name}{Direction}Kernel` (Kernel layer); direction suffix mandatory. Manifest author chooses `{Name}`. Builder functions stay snake_case.

- `kernel_types` is the Op→Kernel dispatch registration table: snake_case dispatch keys (decoupled from class names) → Kernel classes. `default_kernel_map` is derived from it. The code owns it; the manifest does not list kernels. See [op-slot-rules.md § Slot S14](../../docs/design/op-slot-rules.md#slot-s14).

- Op `__init__` takes `signature.params` in manifest order, a param declaring `kw_only: true` after `*`, then the execution-policy parameters of [manifest.md table 7](../../docs/design/manifest.md#t-policy), keyword-only. Every other index is solved per call.

- The call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the manifest entry; an op does not hand-write them or repeat their checks in `forward`.

- Update `docs/design/` when a change alters a top-level decision (an intermediate base class, the kernel-dispatch pattern, a contract between modules); a class attribute or other mechanism that implements a documented decision is read from the code.

- Declare `slots` on every op that holds kernels: slot name → the `Slot` interface its candidates implement. Open a new slot only where semantic control flow or the kernel call contract changes, never per shape, dtype, architecture or performance. See [ops-design.md § Kernel selection](../../docs/design/ops-design.md#kernel-selection).

- Define a slot interface in the family's `kernels/<family>/call_spec.py`: `request` names the frozen `CallSpec` request key, and an abstract `forward` states each tensor's shape, dtype, layout, device, in-place writes and aliasing, and the return value.

- Make each candidate a `Kernel` subclass that inherits its slot interface, takes the interface's `forward` arguments by the same names and positions, and is built only through its classmethod `entry_for(call)`, which returns a hashable build identity and a builder.

- State a candidate's region positively in `applies` / `refusal`. Where its region nests in a sibling's, declare `refines = frozenset({"<sibling key>"})` on it; never exclude a sibling inside a region. Mark at most one candidate per slot `general`.

- Give a shape-selected change of decomposition or data flow its own candidate class. Keep tile sizes, split counts (one included) and fusion among one fixed set of stages inside one candidate's plan.

- Put call facts only in the request key: shapes, dtypes, the relevant layout, semantic params and flags (the op's fixed params included) and `device=`. Never put `tune`, a backend choice, a priority or a device fact (`arch`, `sm_count`, `calibration`) there; the dispatcher resolves device facts on a miss.

- Add a candidate for part of a slot from a backend with `tileops.backend.register_candidate(op, slot, key, cls)`, declaring `refines` where its region nests in an in-tree candidate's.

- Declare `slots` on a new op; `tests/test_slot_dispatch.py` lists the ops still on the unslotted path, and a migration PR removes the names it migrates.

- Every kernel an op builds after construction goes through `Op.kernel_for(slot, inputs, call)`, with `inputs` the tensors the kernel will be handed and `call` the slot's request key. The identity and builder come from the selected candidate's `entry_for`; a slotted op defines no `entry_for` of its own. An op MUST NOT declare a kernel cache dict, guard a kernel build on an attribute being unset, or carry any other get-or-build of its own — including for an auxiliary kernel. Assigning what `kernel_for` returned to `self.kernel` is not one. See [ops-design.md § Kernel caching and enumeration](../../docs/design/ops-design.md#kernel-caching-and-enumeration).

- An op that runs kernels built by another op declares that op's class in `delegate_types` and holds it through `delegate_for(stage, key, ...)`, whether it is built at construction, built per call, or injected by the caller. `kernel_delegates()` is derived and not overridden. A sub-op cache of an op's own, or overriding `autotune()` to reach a delegate, is prohibited.

- A compile-boundary op's `forward` only chooses which operator to call. The generated checks run in the operator's eager body before either implementation; kernel resolution goes in `_eager_forward`. **Why:** a target serving the op is called inside the operator. See [ops-design.md § Target boundary](../../docs/design/ops-design.md#target-boundary).

- A sub-op that depends on the call is built in `_eager_forward`, never in a traced `forward`.

- `__init__` MUST NOT read any device property, directly or through `dispatch_kernel`. An op constructs wherever it is imported; a target that cannot run it is refused when a kernel is first selected, built or called.

- A new op family inheriting `Op` directly: first check whether an existing family's `forward()` flow already fits before creating a new base class. Record the decision in the PR.

- Per-op workarounds MUST NOT be promoted to a base-class shared mechanism (mixin, class attribute, shared method, opt-out flag) within the same op-family migration PR — even when multiple ops share the workaround. Promote only via a separate design PR that shows the mechanism is a genuine family invariant (would belong in the base even if no op had taken a shortcut), not a shared shortcut.

- PyTorch fallback at forward time is permitted only when TileLang cannot express the operation at the required shape AND no closed-form replacement exists in tensor primitives; document the call site with the blocking limitation and a tracking issue. Helper conveniences (`x.float().mean(...)` for clarity) are out of scope — the rule targets full-operator delegation.

- Dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder; call-time kernel resolution goes through the compile dispatch boundary. See [ops-design.md](../../docs/design/ops-design.md#compile-dispatch-boundary).

- A `@tilelang.jit` builder MUST close over scalars only; an op body or a stride tuple goes in a registry, and the builder closes over its name. **Why:** TileLang reads every free variable into the autotune cache key, asserts on anything but `int` / `float` / `str` / `bool` / `None`, and tells two tuned kernels apart by that name.

- A candidate config key MUST name a parameter of the builder being tuned; a parameter spelled `<key>_arg` MUST have `<key>` in `_AUTOTUNE_PARAM_ALIASES`. **Why:** TileLang binds candidates by parameter name and raises on a key that names none.

- A kernel whose integer tensor inputs decide how much work it runs supplies them through `autotune_supply_prog`; one whose integer inputs are data or masks sets `autotune_accepts_random_int_inputs = True` with the reason. `tune_jit_kernel` refuses the unanswered case. **Why:** TileLang generates an unsupplied integer tensor from `randint(-2, 3)`, so every candidate times a collapsed kernel.
