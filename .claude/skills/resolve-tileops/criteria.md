### 1. Inline reply format

- **Accept**: `Adopted. <what was fixed>. See <short_sha>.`
- **Reject**: `<conclusion>. <evidence or reasoning>.`
- **Defer**: `Valid point. Out of scope for this PR — <reason>. Will address in follow-up.`

One reply per thread. Resolve every thread that was replied to, regardless of verdict.

### 2. Top-level note (optional, only when something cross-cutting can't fit inline)

```
Addressed in <short_sha>. See thread replies for specifics.

### Cross-cutting (optional)
- <one line per pattern handled jointly across threads>

### Deferred (optional)
- <one line per item out of scope, with reason>
```

### 3. Hard rules

- Do NOT restate inline replies. If a thread already has the reply, the top-level note must not repeat it.
- No per-file / per-line bullets. Those live inline.
- No GitHub review IDs (`#4181055953`) — meaningless to humans.
- One short paragraph + at most two markdown sections in any top-level note. If you can't fit it, rethink whether it belongs inline.
- All replies in English; concise — conclusion, action, reasoning. No filler.
