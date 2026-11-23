#!/usr/bin/env python3
"""Sync the GitHub Project "Nightly Perf Tracker" with the nightly's perf gap queue.

The queue is ``perf_tracker.build_queue``'s, the one the wiki's Perf Tracker page shows; the
regressed ops are those of the snapshot's ``regressions.json``, rows the nightly report found
slower than their 14-day median. The project holds one item per op, titled by the op.

This script writes the measured fields of the queued ops. It sets an item that is new,
archived, Done, Unclaimed, or without a status to Regressed when the op regresses, else to
Unclaimed while it is queued; moves a caught-up op to Done; and archives items Done for
``ARCHIVE_DAYS``. A Regressed item stays until a maintainer moves it, because a slowdown
that persists becomes its own 14-day median. The other statuses, assignees, and item
bodies are edited by hand.
Calls GraphQL through ``gh``, which reads its token from ``GH_TOKEN``.
"""

import argparse
import json
import subprocess
import sys
from datetime import date as date_type
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_tracker import build_queue, caught_up, load

OWNER, NUMBER = "tile-ai", 7
ARCHIVE_DAYS = 14

PROJECT_QUERY = """
query($owner: String!, $number: Int!) {
  organization(login: $owner) { projectV2(number: $number) {
    id
    fields(first: 50) { nodes {
      ... on ProjectV2FieldCommon { id name dataType }
      ... on ProjectV2SingleSelectField { options { id name } }
    } }
  } }
}"""
ITEMS_QUERY = """
query($owner: String!, $number: Int!, $after: String) {
  organization(login: $owner) { projectV2(number: $number) {
    items(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id isArchived fieldValues(first: 50) { nodes {
        ... on ProjectV2ItemFieldSingleSelectValue { name field { ... on ProjectV2FieldCommon { name } } }
        ... on ProjectV2ItemFieldNumberValue { number field { ... on ProjectV2FieldCommon { name } } }
        ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2FieldCommon { name } } }
        ... on ProjectV2ItemFieldDateValue { date field { ... on ProjectV2FieldCommon { name } } }
      } } }
    }
  } }
}"""
ADD_ITEM = """
mutation($project: ID!, $title: String!) {
  addProjectV2DraftIssue(input: {projectId: $project, title: $title}) { projectItem { id } }
}"""
ARCHIVE = {
    True: "mutation($project: ID!, $item: ID!) {"
    " archiveProjectV2Item(input: {projectId: $project, itemId: $item}) { item { id } } }",
    False: "mutation($project: ID!, $item: ID!) {"
    " unarchiveProjectV2Item(input: {projectId: $project, itemId: $item}) { item { id } } }",
}
SET_FIELD = """
mutation($project: ID!, $item: ID!, $field: ID!, $value: ProjectV2FieldValue!) {
  updateProjectV2ItemFieldValue(
    input: {projectId: $project, itemId: $item, fieldId: $field, value: $value}
  ) { projectV2Item { id } }
}"""


def plan(board: dict, queue: list[dict], caught: set, date: str, regressed: set) -> dict:
    """``{op: {field: value}}`` that brings *board* to tonight's *queue*, *caught* and
    *regressed* ops.

    *board* maps each item's title to its field values, with ``Archived`` among them.
    An op in the queue or regressed without an item gets one; an archived one is
    unarchived. *caught* holds the ops ``caught_up`` finds. Only a caught-up op that is not
    Regressed moves to Done, and only a caught-up op Done for ``ARCHIVE_DAYS`` is archived.
    ``Archived`` comes first in a change.
    """
    # The statuses this script sets; the others are set by hand.
    set_here = (None, "Done", "Unclaimed", "Regressed")
    held = regressed | {
        op
        for op, fields in board.items()
        if fields.get("Status") == "Regressed" and not fields.get("Archived")
    }
    queued = {entry["op"]: entry for entry in queue}
    changes = {}
    for op in queued.keys() | regressed:
        fields = board.get(op) or {}
        want = {"Archived": False} if fields.get("Archived") else {}
        if op in queued:
            entry = queued[op]
            want |= {
                "Priority": entry["label"],
                "Worst ratio": round(entry["worst"], 4),
                "Rows behind": len(entry["rows"]),
                "Nightlies behind": entry["streak"],
                "Category": entry["rows"][0]["category"],
                "Measured": date,
            }
        if want.get("Archived") is False or fields.get("Status") in set_here:
            want["Status"] = "Regressed" if op in held else "Unclaimed"
        diff = {name: value for name, value in want.items() if fields.get(name) != value}
        if op not in board or diff:
            changes[op] = diff
    cutoff = (date_type.fromisoformat(date) - timedelta(days=ARCHIVE_DAYS)).isoformat()
    for op, fields in board.items():
        if fields.get("Archived") or op in held or op not in caught:
            continue
        if fields.get("Status") != "Done" or not fields.get("Measured"):
            changes[op] = {"Status": "Done", "Measured": date}
        elif fields["Measured"] <= cutoff:
            changes[op] = {"Archived": True}
    return changes


def read_regressions(repo: Path, commit: str) -> set:
    """The ops of a snapshot's ``regressions.json``; none when the snapshot has no such file."""
    git = ["git", "-C", str(repo)]
    files = subprocess.check_output([*git, "ls-tree", "--name-only", commit], text=True)
    if "regressions.json" not in files.split():
        return set()
    out = subprocess.check_output([*git, "show", f"{commit}:regressions.json"])
    return {record["op"] for record in json.loads(out)}


def _graphql(query: str, **variables) -> dict:
    body = json.dumps({"query": query, "variables": variables})
    out = subprocess.check_output(["gh", "api", "graphql", "--input", "-"], input=body, text=True)
    return json.loads(out)["data"]


def read_board() -> tuple[str, dict, dict]:
    """The project's id, its fields by name, and ``{title: (item id, {field: value})}``."""
    project = _graphql(PROJECT_QUERY, owner=OWNER, number=NUMBER)["organization"]["projectV2"]
    fields = {f["name"]: f for f in project["fields"]["nodes"] if f}
    items, after = {}, None
    while True:
        page = _graphql(ITEMS_QUERY, owner=OWNER, number=NUMBER, after=after)
        page = page["organization"]["projectV2"]["items"]
        for node in page["nodes"]:
            values = {
                v["field"]["name"]: next(x for k, x in v.items() if k != "field")
                for v in node["fieldValues"]["nodes"]
                if v.get("field")
            }
            title = values.get("Title")
            if title in items:
                raise ValueError(f"two project items are titled {title}")
            items[title] = (node["id"], {**values, "Archived": node["isArchived"]})
        if not page["pageInfo"]["hasNextPage"]:
            return project["id"], fields, items
        after = page["pageInfo"]["endCursor"]


def write_board(project: str, fields: dict, items: dict, changes: dict) -> None:
    for op, values in changes.items():
        item = items[op][0] if op in items else None
        if item is None:
            item = _graphql(ADD_ITEM, project=project, title=op)
            item = item["addProjectV2DraftIssue"]["projectItem"]["id"]
        for name, value in values.items():
            if name == "Archived":
                _graphql(ARCHIVE[value], project=project, item=item)
                continue
            field = fields[name]
            if field["dataType"] == "SINGLE_SELECT":
                value = {
                    "singleSelectOptionId": {o["name"]: o["id"] for o in field["options"]}[value]
                }
            else:
                value = {field["dataType"].lower(): value}
            _graphql(SET_FIELD, project=project, item=item, field=field["id"], value=value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path, help="TileOPs-nightly clone")
    parser.add_argument("--run-id", required=True, help="the nightly run that must be newest")
    parser.add_argument("--date", required=True, help="the Measured date, YYYY-MM-DD")
    parser.add_argument("--nights", type=int, default=30, help="nightlies judged for the streaks")
    parser.add_argument("--dry-run", action="store_true", help="print the changes only")
    args = parser.parse_args()

    runs, rows_by_run = load(args.snapshots, args.run_id, args.nights)
    queue, behind, _ = build_queue(rows_by_run)
    regressed = read_regressions(args.snapshots, runs[0]["commit"])
    project, fields, items = read_board()
    board = {title: values for title, (_, values) in items.items()}
    caught = caught_up(rows_by_run, behind)
    changes = plan(board, queue, caught, args.date, regressed)
    for op, values in changes.items():
        print(f"{op}{'' if op in items else ' (new)'}: {json.dumps(values)}")
    if not args.dry_run:
        write_board(project, fields, items, changes)
    new = len(changes.keys() - items.keys())
    print(f"{len(changes) - new} items changed, {new} added{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
