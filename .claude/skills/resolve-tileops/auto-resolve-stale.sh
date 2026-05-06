#!/usr/bin/env bash
# auto-resolve-stale.sh — classify unresolved review threads for the
# stale-bot auto-resolution policy and (in normal mode) execute the
# GraphQL mutations that auto-reply + resolve qualifying threads.
#
# Inputs:
#   --threads <file>   JSON {head_sha, threads:[{id, comments:{nodes:[
#                      {id, databaseId, author:{login}, commit:{oid},
#                       body, path, line}]}}]}
#   --bots <file>      known-bots.json
#   --run-dir <dir>    where to drop unknown-bot-like artifact
#   --round <NN>       zero-padded round number for artifact filename
#   --dry-run          do not call the GraphQL API; just emit the action plan
#
# Stdout: JSON {resolve:[{thread_id,comment_id,login,reply}],
#               unknown_bot_like:[{thread_id,login}],
#               skip:[{thread_id,reason}]}
#
# Side effect: $RUN_DIR/round-<NN>.unknown-bot-like.json (only when
# at least one unknown bot-like thread was seen this call).

set -euo pipefail

REPLY_TEXT="Not assessed on latest HEAD"

THREADS_FILE=""; BOTS_FILE=""; RUN_DIR=""; ROUND=""; DRY_RUN=0
while (( $# )); do
  case "$1" in
    --threads)  THREADS_FILE="$2"; shift 2 ;;
    --bots)     BOTS_FILE="$2"; shift 2 ;;
    --run-dir)  RUN_DIR="$2"; shift 2 ;;
    --round)    ROUND="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    *) echo "auto-resolve-stale: unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -f "$THREADS_FILE" ]] || { echo "auto-resolve-stale: missing --threads" >&2; exit 2; }
[[ -f "$BOTS_FILE"    ]] || { echo "auto-resolve-stale: missing --bots"    >&2; exit 2; }
[[ -n "$RUN_DIR"      ]] || { echo "auto-resolve-stale: missing --run-dir" >&2; exit 2; }
[[ "$ROUND" =~ ^[0-9]{2}$ ]] || { echo "auto-resolve-stale: --round must be zero-padded 2-digit" >&2; exit 2; }

command -v jq >/dev/null 2>&1 || { echo "auto-resolve-stale: missing jq" >&2; exit 2; }

# Build the action plan in pure jq — single pass, no shell-side per-thread state.
# REPLY_TEXT is passed in as --arg so the action plan and the executed
# mutation body below share a single source of truth.
# Validate the unknown-bot-like policy up front so the config field is
# load-bearing rather than decorative. Only "log_for_manual_triage" is
# implemented today — any other value fails fast.
POLICY=$(jq -r '.policy.unknown_bot_like_login // ""' "$BOTS_FILE")
if [[ "$POLICY" != "log_for_manual_triage" ]]; then
  echo "auto-resolve-stale: unsupported policy.unknown_bot_like_login='$POLICY' (expected 'log_for_manual_triage')" >&2
  exit 2
fi

PLAN=$(jq --slurpfile bots "$BOTS_FILE" --arg reply "$REPLY_TEXT" '
  # Normalise both sides: strip a trailing "[bot]" suffix before comparing.
  # GitHub returns either "copilot-pull-request-reviewer" or
  # "copilot-pull-request-reviewer[bot]" depending on the API; treat them
  # as the same identity for the whitelist check.
  def strip_bot: sub("\\[bot\\]$"; "");
  def is_known($known; $login):
    ($known | map(strip_bot) | index($login | strip_bot)) != null;
  # Bot-like pattern: unknown identities that look like a GitHub App.
  # Only the literal "[bot]" suffix counts — that suffix is reserved by
  # GitHub for GitHub Apps and cannot appear in a human username. Any
  # login ending in "[bot]" that is not on the whitelist is bucketed as
  # an unknown bot-like identity for human triage, regardless of the
  # prefix. A bare "-reviewer" / "-bot" without the suffix is a regular
  # user account (e.g. "alice-reviewer") and stays in the human bucket.
  def is_bot_like($login):
      $login | test("\\[bot\\]$");
  . as $in
  | ($bots[0].review_bot_logins // []) as $known
  | ($in.head_sha) as $head
  | [ $in.threads[]
      | . as $t
      | ($t.comments.nodes[0]) as $first
      | ($first.author.login // "") as $login
      | ($first.commit.oid // "") as $oid
      # Whole-thread author check: if ANY comment in the thread is
      # authored by a human (i.e. a login that is neither whitelisted
      # nor matches the GitHub-App "[bot]" suffix), treat the thread as
      # having human discussion and never auto-resolve. Threads whose
      # only authors are known bots — or known bots plus other bot-like
      # apps — remain bot-only for classification purposes.
      | ([ $t.comments.nodes[]
            | (.author.login // "")
            | select(
                . != ""
                and (is_known($known; .) | not)
                and (is_bot_like(.)       | not)
              )
         ] | length) as $human_repliers
      | {
          thread_id: $t.id,
          comment_id: ($first.id // ""),
          login: $login,
          oid: $oid,
          known_bot: is_known($known; $login),
          bot_like:  is_bot_like($login),
          # Distinct buckets:
          #   missing_oid: comment has no commit anchor → cannot judge stale
          #   stale: oid present and != head
          #   at_head: oid present and == head
          missing_oid: ($oid == ""),
          stale:       ($oid != $head and $oid != ""),
          mixed:       ($human_repliers > 0)
        }
    ] as $rows
  | {
      resolve: [
        $rows[] | select(.known_bot and .stale and (.mixed | not))
        | { thread_id, comment_id, login, reply: $reply }
      ],
      unknown_bot_like: [
        $rows[] | select((.known_bot|not) and .bot_like and (.mixed | not))
        | { thread_id, login }
      ],
      skip: [
        $rows[]
        | select(
            .mixed                                       # any human/unknown-bot replier in thread
            or ((.known_bot|not) and (.bot_like|not))    # human-rooted
            or (.known_bot and (.stale|not))             # bot not stale (at HEAD or missing oid)
          )
        | {
            thread_id,
            reason: (
              if .mixed and .known_bot then "mixed_thread_known_bot_root"
              elif .mixed and .bot_like then "mixed_thread_unknown_bot_like_root"
              elif .mixed then "mixed_thread_human_root"
              elif .known_bot and .missing_oid then "known_bot_missing_commit_oid"
              elif .known_bot and (.stale|not) then "known_bot_at_head"
              elif (.bot_like|not) then "human_reviewer"
              else "other"
              end
            )
          }
      ]
    }
' "$THREADS_FILE")

# Drop the artifact for human triage when there is anything to record.
ARTIFACT="$RUN_DIR/round-${ROUND}.unknown-bot-like.json"
UNKNOWN_COUNT=$(printf '%s' "$PLAN" | jq '.unknown_bot_like | length')
if (( UNKNOWN_COUNT > 0 )); then
  mkdir -p "$RUN_DIR"
  printf '%s' "$PLAN" | jq '.unknown_bot_like' > "$ARTIFACT"
fi

if (( DRY_RUN )); then
  printf '%s\n' "$PLAN"
  exit 0
fi

# Execute mutations for each resolve entry: post a reply on the first
# comment, then — only if the reply succeeded — mark the thread resolved.
# Skipping the resolve when the reply fails preserves the contract that
# every auto-resolved thread carries the neutral reply, so a thread we
# could not reply to is left unresolved for the next round / human
# triage. Per-thread failures are logged but do not abort the loop, so
# one transient API hiccup doesn't strand the rest of the batch.
#
# GH_BIN allows the test harness to inject a mock gh binary that
# simulates reply / resolve failures without touching the network.
GH_BIN="${GH_BIN:-gh}"
command -v "$GH_BIN" >/dev/null 2>&1 || { echo "auto-resolve-stale: missing $GH_BIN" >&2; exit 2; }
RESOLVE_COUNT=$(printf '%s' "$PLAN" | jq '.resolve | length')
RESOLVED_IDS=()
REPLY_FAILED_IDS=()
RESOLVE_FAILED_IDS=()
if (( RESOLVE_COUNT > 0 )); then
  while IFS= read -r tid; do
    [[ -n "$tid" ]] || continue
    if "$GH_BIN" api graphql -f query='
      mutation($tid:ID!,$body:String!){
        addPullRequestReviewThreadReply(input:{
          pullRequestReviewThreadId:$tid, body:$body
        }){ comment{ id } }
      }' -F tid="$tid" -F body="$REPLY_TEXT" >/dev/null; then
      if "$GH_BIN" api graphql -f query='
        mutation($tid:ID!){
          resolveReviewThread(input:{threadId:$tid}){ thread{ id isResolved } }
        }' -F tid="$tid" >/dev/null; then
        RESOLVED_IDS+=("$tid")
      else
        echo "auto-resolve-stale: resolve failed for $tid" >&2
        RESOLVE_FAILED_IDS+=("$tid")
      fi
    else
      echo "auto-resolve-stale: reply failed for $tid; leaving thread unresolved" >&2
      REPLY_FAILED_IDS+=("$tid")
    fi
  done < <(printf '%s' "$PLAN" | jq -r '.resolve[].thread_id')
fi

# Annotate the plan with execution outcomes so the caller (and the test
# harness) can distinguish "resolved" from "reply failed, left open".
# Use `jq -n '$ARGS.positional' --args …` to lift each shell array into
# a JSON string array in one shot — the `${arr[@]+"${arr[@]}"}` guard is
# needed under `set -u` so an empty array does not trip "unbound
# variable".
printf '%s' "$PLAN" | jq \
  --argjson resolved       "$(jq -n '$ARGS.positional' --args ${RESOLVED_IDS[@]+"${RESOLVED_IDS[@]}"})" \
  --argjson reply_failed   "$(jq -n '$ARGS.positional' --args ${REPLY_FAILED_IDS[@]+"${REPLY_FAILED_IDS[@]}"})" \
  --argjson resolve_failed "$(jq -n '$ARGS.positional' --args ${RESOLVE_FAILED_IDS[@]+"${RESOLVE_FAILED_IDS[@]}"})" \
  '. + {executed: {resolved: $resolved, reply_failed: $reply_failed, resolve_failed: $resolve_failed}}'
