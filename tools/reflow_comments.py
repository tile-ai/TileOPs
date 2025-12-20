#!/usr/bin/env python3
"""Simple reflow for leading-comment blocks in Python files.

This script only reflows comment-only blocks (lines where the first
non-space character is '#'). It preserves indentation and the '# '
prefix. It does not touch inline comments (after code) or string
literals.

Usage: python tools/reflow_comments.py file1.py file2.py ...
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

COLUMN = 100


def reflow_file(path: Path, width: int = COLUMN) -> bool:
    text = path.read_text(encoding="utf8")
    lines = text.splitlines()
    out_lines = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # skip shebang and lines that are not pure leading-comments
        if stripped.startswith("#!") or not stripped.startswith("#"):
            out_lines.append(line)
            i += 1
            continue

        # collect contiguous comment-only block
        block = []
        indent = line[:len(line) - len(stripped)]
        while i < len(lines):
            s = lines[i]
            s_stripped = s.lstrip()
            if not s_stripped.startswith("#"):
                break
            # remove leading '#' and one optional single space after it
            content = s_stripped[1:]
            if content.startswith(" "):
                content = content[1:]
            block.append(content)
            i += 1

        # join block paragraphs separated by empty comment lines
        paragraphs = []
        para = []
        for b in block:
            if b.strip() == "":
                if para:
                    paragraphs.append(" ".join(p.strip() for p in para))
                    para = []
                else:
                    paragraphs.append("")
            else:
                para.append(b)
        if para:
            paragraphs.append(" ".join(p.strip() for p in para))

        # reflow each paragraph and emit
        for _pi, p in enumerate(paragraphs):
            if p == "":
                out_lines.append(indent + "#")
                continue
            wrap_width = max(20, width - len(indent) - 2)
            wrapped = textwrap.fill(
                p,
                width=wrap_width,
                replace_whitespace=True,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for wl in wrapped.splitlines():
                out_lines.append(f"{indent}# {wl}")

        # detect if content changed
        original_block_lines = []
        for b in block:
            if b == "":
                original_block_lines.append(indent + "#")
            else:
                original_block_lines.append(indent + "# " + b.rstrip())
        if out_lines[-len(original_block_lines):] != original_block_lines:
            changed = True

    if changed:
        path.write_text(
            "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf8")
    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: reflow_comments.py <files...>")
        return 1
    any_changed = False
    for fname in argv[1:]:
        p = Path(fname)
        if not p.exists():
            continue
        try:
            changed = reflow_file(p)
            if changed:
                print(f"Reflowed: {p}")
                any_changed = True
        except Exception as e:
            print(f"Error processing {p}: {e}", file=sys.stderr)
    return 1 if any_changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
