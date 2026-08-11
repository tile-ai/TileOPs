→ [trust-model.md §Implementation](../../docs/design/trust-model.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

- Class names: PascalCase `{Name}{Direction}Op` (Op layer) or `{Name}{Direction}Kernel` (Kernel layer); direction suffix mandatory. Manifest author chooses `{Name}`. Builder functions stay snake_case.

- `kernel_map` is the Op→Kernel dispatch registration table: snake_case dispatch keys (decoupled from class names) → Kernel class names. Manifest declares it; agents implement the listed Kernels. See [scaffold-op § Slot S14](../skills/scaffold-op/slot-rules.md#slot-s14).

- Op `__init__` is keyword-only (`def __init__(self, *, ...)`). Parameter names come from the manifest: `shape` dim names (fixed-rank), `static_dims` keys (arbitrary-rank), `params` keys. Only manifest-declared information belongs in `__init__`.

- Arbitrary-rank ops declare construction-time values via manifest `static_dims`. Each entry is a single-axis reference `<tensor>.shape[<const_or_param>]`; other dims come from tensors at forward time. See [manifest.md R20](../../docs/design/manifest.md).

- Update `docs/design/ops-design.md` whenever you add/modify an intermediate base class, change a kernel-dispatch pattern, or introduce a new class-variable protocol.

- Every kernel an op builds after construction goes through `Op.get_or_build_kernel(name, inputs, *, key, build)` — `inputs` is what the kernel will be handed, so an external target can be asked to build one instead. An op MUST NOT declare a kernel cache dict, guard a kernel build on an attribute being unset, or carry any other get-or-build of its own — including for an auxiliary kernel. Assigning what `get_or_build_kernel` returned to `self.kernel` is not one. See [ops-design.md § Kernel caching and enumeration](../../docs/design/ops-design.md#kernel-caching-and-enumeration).

- An op that runs kernels built by another op returns that op from `kernel_delegates()`, whether the delegate is fixed at construction or built per specialization. Overriding `autotune()` to reach a delegate, or exposing a delegate's cache so reflection finds it, is prohibited.

- `__init__` MUST NOT read any device property, directly or through `dispatch_kernel`. An op constructs wherever it is imported; a target that cannot run it is refused when a kernel is first selected, built or called.

- A new op family inheriting `Op` directly: first check whether an existing family's `forward()` flow already fits before creating a new base class. Record the decision in the PR.

- Per-op workarounds MUST NOT be promoted to a base-class shared mechanism (mixin, class attribute, shared method, opt-out flag) within the same op-family migration PR — even when multiple ops share the workaround. Promote only via a separate design PR that shows the mechanism is a genuine family invariant (would belong in the base even if no op had taken a shortcut), not a shared shortcut.

- PyTorch fallback at forward time is permitted only when TileLang cannot express the operation at the required shape AND no closed-form replacement exists in tensor primitives; document the call site with the blocking limitation and a tracking issue. Helper conveniences (`x.float().mean(...)` for clarity) are out of scope — the rule targets full-operator delegation.

- Inline roofline state contract: for every `signature.inputs` / `signature.params` name **referenced** by the op's manifest `roofline` expressions, the op exposes it on `self`. Inputs: `self.<input>` with `.shape` and `.ndim`, OR `self.<input>_shape` as a shape tuple/list. Params: `self.<param>`. Unreferenced names need not be exposed. See [docs/design/roofline.md §4.4.3](../../docs/design/roofline.md).

- Dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder; call-time kernel resolution goes through the compile dispatch boundary. See [ops-design.md](../../docs/design/ops-design.md#compile-dispatch-boundary).
