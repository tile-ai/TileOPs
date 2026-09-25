→ [trust-model.md §Implementation](../../docs/design/trust-model.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

- Class names: PascalCase `{Name}{Direction}Op` (Op layer) or `{Name}{Direction}Kernel` (Kernel layer); direction suffix mandatory. Manifest author chooses `{Name}`. Builder functions stay snake_case.

- `default_kernel_map` is the Op→Kernel dispatch registration table: snake_case dispatch keys (decoupled from class names) → Kernel class names. The code owns it; the manifest does not list kernels. See [op-slot-rules.md § Slot S14](../../docs/design/op-slot-rules.md#slot-s14).

- Op `__init__` takes `signature.params` in manifest order, a param declaring `kw_only: true` after `*`, then the execution-policy parameters of [manifest.md table 7](../../docs/design/manifest.md#t-policy), keyword-only. Every other index is solved per call.

- The call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the manifest entry; an op does not hand-write them or repeat their checks in `forward`.

- Update `docs/design/ops-design.md` whenever you add/modify an intermediate base class, change a kernel-dispatch pattern, or introduce a new class-variable protocol.

- Every kernel an op builds after construction goes through `Op.kernel_for(role, inputs, call)`, with `inputs` the tensors the kernel will be handed. The in-tree identity and builder come from `Op.entry_for(role, call)`, whose default selects among the op's candidates and asks the chosen class; an op with one implementation overrides it. An op MUST NOT declare a kernel cache dict, guard a kernel build on an attribute being unset, or carry any other get-or-build of its own — including for an auxiliary kernel. Assigning what `kernel_for` returned to `self.kernel` is not one. See [ops-design.md § Kernel caching and enumeration](../../docs/design/ops-design.md#kernel-caching-and-enumeration).

- An op that runs kernels built by another op returns that op from `kernel_delegates()`, whether the delegate is fixed at construction or built per specialization. Overriding `autotune()` to reach a delegate, or exposing a delegate's cache so reflection finds it, is prohibited.

- A compile-boundary op's `forward` only chooses which operator to call. The generated checks run in the operator's eager body before either implementation; kernel resolution goes in `_eager_forward`. **Why:** a target serving the op is called inside the operator. See [ops-design.md § Target boundary](../../docs/design/ops-design.md#target-boundary).

- A composite passes its `target` to every sub-op it builds, including one built lazily.

- `__init__` MUST NOT read any device property, directly or through `dispatch_kernel`. An op constructs wherever it is imported; a target that cannot run it is refused when a kernel is first selected, built or called.

- A new op family inheriting `Op` directly: first check whether an existing family's `forward()` flow already fits before creating a new base class. Record the decision in the PR.

- Per-op workarounds MUST NOT be promoted to a base-class shared mechanism (mixin, class attribute, shared method, opt-out flag) within the same op-family migration PR — even when multiple ops share the workaround. Promote only via a separate design PR that shows the mechanism is a genuine family invariant (would belong in the base even if no op had taken a shortcut), not a shared shortcut.

- PyTorch fallback at forward time is permitted only when TileLang cannot express the operation at the required shape AND no closed-form replacement exists in tensor primitives; document the call site with the blocking limitation and a tracking issue. Helper conveniences (`x.float().mean(...)` for clarity) are out of scope — the rule targets full-operator delegation.

- Dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder; call-time kernel resolution goes through the compile dispatch boundary. See [ops-design.md](../../docs/design/ops-design.md#compile-dispatch-boundary).

- A `@tilelang.jit` builder MUST close over scalars only; an op body or a stride tuple goes in a registry, and the builder closes over its name. **Why:** TileLang reads every free variable into the autotune cache key, asserts on anything but `int` / `float` / `str` / `bool` / `None`, and tells two tuned kernels apart by that name.

- A candidate config key MUST name a parameter of the builder being tuned; a parameter spelled `<key>_arg` MUST have `<key>` in `_AUTOTUNE_PARAM_ALIASES`. **Why:** TileLang binds candidates by parameter name and raises on a key that names none.

- A kernel whose integer tensor inputs decide how much work it runs supplies them through `autotune_supply_prog`; one whose integer inputs are data or masks sets `autotune_accepts_random_int_inputs = True` with the reason. `tune_jit_kernel` refuses the unanswered case. **Why:** TileLang generates an unsupplied integer tensor from `randint(-2, 3)`, so every candidate times a collapsed kernel.
