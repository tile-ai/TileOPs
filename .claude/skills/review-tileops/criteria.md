# 1. Submit

```bash
gh api repos/tile-ai/TileOPs/pulls/<N>/reviews \
  -f event="<EVENT>" \
  -f body="<SUMMARY>" \
  -f 'comments=[{"path":"<file>","line":<line>,"body":"<comment>"}, ...]'
```

`<EVENT>`: `REQUEST_CHANGES` if any blocking issue, `APPROVE` if clean, `COMMENT` for non-blocking questions only.

# 2. Inline format

```
<what is wrong and why> → <what to change>
```

One comment per issue. Name the function, variable, or pattern. The reader is an agent that executes fixes literally.

# 3. Summary format (markdown)

The summary is for what does **not** fit in an inline comment. Per-file issues belong inline (§2), not in a summary list. Omit empty sections.

```
### Overall
<one line: top risk + next step. No event echo, no item list.>

### Cross-cutting concerns
- <pattern spanning multiple files — only what an inline comment can't carry>
```

Hard rules:

- Do NOT restate inline comments. If a finding is already inline, it must not appear in the summary.
- Per-file/per-line items go in the `comments=[...]` array of §1, never in the summary body.
- Clean PR: one line, `Clean — no issues.`

# 4. Hard rules

- Read every changed file in full — diff alone lacks context.
- Do not comment outside the PR diff.
- Do not invent issues on a clean PR.
- No eager design-doc reads — only when a checklist item names a doc AND the diff makes that item ambiguous.
- All review text in English.
- All `gh` calls run with `GH_CONFIG_DIR` already exported by the caller. Repo is fixed at `tile-ai/TileOPs`.
