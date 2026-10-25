---
name: Bug report
about: Wrong results, a crash, or an error from a TileOPs op
title: '[BUG][<COMPONENT>] <short description in lowercase>'
labels: bug
assignees: ''
---

## What happened

<!-- The op, the call, and the error or the wrong output. -->

## Reproduction

```python
# Smallest script that shows the problem: shapes, dtypes, op parameters.
```

## Expected behavior

## Environment

<!-- Paste the output of:
python -c "import torch, tilelang, tileops; print(tileops.__version__, torch.__version__, torch.version.cuda, tilelang.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
and the output of `nvidia-smi` (driver version). -->
