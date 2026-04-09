"""Tests for scripts/migrate_perf_history.py."""

from __future__ import annotations

import pytest

from scripts.migrate_perf_history import migrate

pytestmark = pytest.mark.smoke


class TestMigrateNestedFormat:
    """Test migration of the real perf_history.json nested format."""

    def test_renames_op_keys_inside_runs(self):
        """The real format is {"runs": [{"ops": {"OldName": {...}}}]}."""
        data = {
            "runs": [
                {
                    "date": "2026-01-01",
                    "commit": "abc123",
                    "gpu": "H100",
                    "ops": {
                        "SoftmaxOp": {"default": {"tileops": {"latency_ms": 0.5}}},
                        "SumOp": {"default": {"tileops": {"latency_ms": 0.3}}},
                    },
                },
                {
                    "date": "2026-01-02",
                    "commit": "def456",
                    "gpu": "H100",
                    "ops": {
                        "SoftmaxOp": {"default": {"tileops": {"latency_ms": 0.4}}},
                        "LayerNormOp": {"default": {"tileops": {"latency_ms": 1.2}}},
                    },
                },
            ]
        }
        migrated, count = migrate(data)

        # Structure preserved
        assert "runs" in migrated
        assert len(migrated["runs"]) == 2

        # Run 0: both ops renamed
        ops0 = migrated["runs"][0]["ops"]
        assert "SoftmaxFwdOp" in ops0
        assert "SumFwdOp" in ops0
        assert "SoftmaxOp" not in ops0
        assert "SumOp" not in ops0

        # Run 1: both ops renamed
        ops1 = migrated["runs"][1]["ops"]
        assert "SoftmaxFwdOp" in ops1
        assert "LayerNormFwdOp" in ops1
        assert "SoftmaxOp" not in ops1
        assert "LayerNormOp" not in ops1

        # 4 total renames (2 in run 0 + 2 in run 1)
        assert count == 4

    def test_preserves_run_metadata(self):
        """Non-ops fields in each run entry must be preserved."""
        data = {
            "runs": [
                {
                    "date": "2026-03-15",
                    "commit": "xyz789",
                    "gpu": "A100",
                    "ops": {
                        "RmsNormOp": {"cfg1": {"tileops": {"latency_ms": 0.7}}},
                    },
                }
            ]
        }
        migrated, count = migrate(data)
        run = migrated["runs"][0]
        assert run["date"] == "2026-03-15"
        assert run["commit"] == "xyz789"
        assert run["gpu"] == "A100"
        assert count == 1

    def test_preserves_op_data_values(self):
        """Op data (config entries, latencies) must not be modified."""
        original_data = {"default": {"tileops": {"latency_ms": 0.5, "tflops": 12.3}}}
        data = {
            "runs": [
                {
                    "date": "2026-01-01",
                    "ops": {"SoftmaxOp": original_data},
                }
            ]
        }
        migrated, _ = migrate(data)
        assert migrated["runs"][0]["ops"]["SoftmaxFwdOp"] == original_data

    def test_unknown_op_keys_unchanged(self):
        """Op keys not in RENAME_MAP are kept as-is."""
        data = {
            "runs": [
                {
                    "date": "2026-01-01",
                    "ops": {
                        "UnknownOp": {"cfg": {"tileops": {"latency_ms": 1.0}}},
                        "SumOp": {"cfg": {"tileops": {"latency_ms": 0.3}}},
                    },
                }
            ]
        }
        migrated, count = migrate(data)
        ops = migrated["runs"][0]["ops"]
        assert "UnknownOp" in ops
        assert "SumFwdOp" in ops
        assert count == 1

    def test_empty_runs_list(self):
        """Empty runs list should work fine."""
        data = {"runs": []}
        migrated, count = migrate(data)
        assert migrated == {"runs": []}
        assert count == 0

    def test_run_without_ops_key(self):
        """A run entry missing the 'ops' key should be preserved without error."""
        data = {
            "runs": [
                {"date": "2026-01-01", "commit": "abc"},
            ]
        }
        migrated, count = migrate(data)
        assert migrated["runs"][0] == {"date": "2026-01-01", "commit": "abc"}
        assert count == 0


class TestMigrateFlatFormat:
    """Test migration of a flat dict format (backward compat)."""

    def test_flat_dict_renames_top_level_keys(self):
        """A flat dict without 'runs' should still rename top-level keys."""
        data = {
            "SoftmaxOp": {"latency": 0.5},
            "SumOp": {"latency": 0.3},
            "AlreadyNewOp": {"latency": 1.0},
        }
        migrated, count = migrate(data)
        assert "SoftmaxFwdOp" in migrated
        assert "SumFwdOp" in migrated
        assert "AlreadyNewOp" in migrated
        assert count == 2

    def test_empty_flat_dict(self):
        data = {}
        migrated, count = migrate(data)
        assert migrated == {}
        assert count == 0
