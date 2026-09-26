→ [layer-boundaries.md §Manifest](../../docs/design/layer-boundaries.md#manifest) | Spec: [manifest.md](../../docs/design/manifest.md)

- Manifest key equals the Op `cls.__name__` exactly.

- The signature is the public contract; write it against the operator's reference semantics, never against current code. When `ref_api` is present it is the semantic oracle. Include every supported parameter even if the kernel only honors the default.

- `inputs`, `outputs` and `params` are ordered: key order is signature position. Don't reorder.

- Every tensor declares `dtype` and `shape`. Name each free axis in `forall`; write a shared shape once (`[*S]`); derive a dimension with `let`, never with an equality in `shape_rules`.

- `shape_rules` hold refinements only. Presence is `present(x)`, never `x is None`; an axis helper is a built-in primitive, never `isinstance`. A constraint on a metadata tensor's contents is its `requires`.

- A parameter that selects a shape is a discriminant of a type family in `signature.types`. A finite choice with fields is an ADT in `types.yaml`.

- An output dtype is a dtype expression: a `forall` `DType` index, a constant, a dtype primitive, or a dtype parameter. `dtype_combos` only when the supported set is a strict subset of the product.

- Mutation and aliasing are declared on tensors (`mutated`, `write_only`, `buffer`, `alias`). Workspaces are not in the manifest.

- Workload rows give construction parameters, the relevant indices no generator solves, `some`, `dtype_cases` and `label`. Metadata values come from `values` generators. In an implemented entry, every optional input has a row passing it and a row omitting it.

- `roofline` gives `flops`; omit `bytes` unless the derived count is wrong, and then add a test.

- `status: spec-only` until an implementation conforms. Never edit the manifest to match non-conforming code, and never delete a rule to silence the validator.

- Comments carry technical content the schema cannot express, never process metadata bound to an issue, PR or round. Scan: `grep -rnE '#[0-9]{3,}|[Ff]ollow.?up|AC-[0-9]+' src/tileops/manifest/*.yaml`
