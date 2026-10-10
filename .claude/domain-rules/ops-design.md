→ [layer-boundaries.md §Implementation](../../docs/design/layer-boundaries.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

- Class names: PascalCase `{Name}{Direction}Op` (Op layer), `{Name}{Direction}Interface` (kernel interface) and `{Name}Kernel` (Kernel layer). The Op and interface direction suffix is mandatory, with variant words before it; a Kernel names its algorithm and variant. Manifest author chooses `{Name}`. Builder functions stay snake_case.

- `kernel_types` is the Op→Kernel dispatch registration table: snake_case dispatch keys (decoupled from class names) → Kernel classes. `default_kernel_map` is derived from it. The code owns it; the manifest does not list kernels. See [op-slot-rules.md § Slot S14](../../docs/design/op-slot-rules.md#slot-s14).

- Op `__init__` takes `signature.params` in manifest order, a param declaring `kw_only: true` after `*`, then the execution-policy parameters of [manifest.md table 7](../../docs/design/manifest.md#t-policy), keyword-only. Every other index is solved per call.

- The call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the manifest entry; an op does not hand-write them or repeat their checks in `forward`.

- Update `docs/design/` when a change alters a top-level decision (an intermediate base class, the kernel-dispatch pattern, a contract between modules); a class attribute or other mechanism that implements a documented decision is read from the code.

- Declare `interfaces` on every op that holds kernels: the name of each place the op calls a kernel → its `KernelInterface` class. Open a new interface only where semantic control flow or the kernel call contract changes, never per shape, dtype, architecture or performance. See [ops-design.md § Kernel selection](../../docs/design/ops-design.md#kernel-selection).

- Define a kernel interface in the family's `kernels/<family>/call_spec.py`, or in the kernel module of a family with one kernel file: `request` names the frozen `CallSpec` subclass; an abstract `forward` states the tensors handed and the value returned.

- Make each implementation a `Kernel` subclass that inherits its interface, takes the interface's `forward` arguments, and is built only through its classmethod `entry_for(call)`, which returns a hashable build identity and a builder.

- State where an implementation runs in `devices` / `supported_archs` only, and the calls it serves positively in `applies` / `refusal`; never exclude a sibling. Where it overlaps another non-general implementation, declare `preferred_over = frozenset({"<key>"})` on the one that wins. Mark at most one implementation per interface `general`.

- Give a shape-selected change of decomposition or data flow its own implementation class. Keep tile sizes, split counts (one included) and fusion among one fixed set of stages inside one implementation's plan.

- An `optional: true` input defaults to `None` in `forward`, and presence is read from the call, not settled at construction. Where the presence changes what the kernel build produces, it goes in that kernel's cache key.

- Put only call facts in the call spec, the op's fixed semantic params and `device=` included; never `tune`, a priority or a device fact (`arch`, `sm_count`, `calibration`, `smem_budget`), which the dispatcher resolves on a miss.

- Add an implementation from a backend with `tileops.backend.register_implementation(op, key, cls)`; its interface is the one `cls` inherits. Use `kernel_map=` only to replace what runs under an existing key: the key keeps its registered implementation's `applies`, `general` and `preferred_over`. A replacement, like a registered implementation, inherits the key's interface and is built through its own classmethod `entry_for(call)`; there is no other form.

- Declare `interfaces` on a new op; `tests/test_kernel_dispatch.py` lists the ops still without them, and a migration PR removes the names it migrates.

- Every kernel an op builds after construction goes through `Op.kernel_for(interface, call)`, with `call` the interface's call spec. The tensors the kernel is handed are the parameters of the interface's abstract `forward`. The identity and builder come from the selected implementation's `entry_for`; an op with `interfaces` defines no `entry_for` of its own. An op MUST NOT declare a kernel cache dict, guard a kernel build on an attribute being unset, or carry any other get-or-build of its own — including for an auxiliary kernel. Assigning what `kernel_for` returned to `self.kernel` is not one. See [ops-design.md § Kernel caching and enumeration](../../docs/design/ops-design.md#kernel-caching-and-enumeration).

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

- Time the candidates `Kernel.tuning_candidates(configs, untuned)` returns in every tuning loop, with *untuned* the launch config the untuned kernel runs; `tune_jit_kernel` does so with its `seed_config`. **Why:** tuning keeps the fastest config it times, so a search space without the untuned config can leave a tuned kernel slower than an untuned one.

- A kernel whose integer tensor inputs decide how much work it runs supplies them through `autotune_supply_prog`; one whose integer inputs are data or masks sets `autotune_accepts_random_int_inputs = True` with the reason. `tune_jit_kernel` refuses the unanswered case. **Why:** TileLang generates an unsupplied integer tensor from `randint(-2, 3)`, so every candidate times a collapsed kernel.
