"""Phase 16 regression tests: legacy code is recoverable but out of the pipeline.

REFACTOR_PLAN.md §21 keeps the state-based workflow and the VariDP donor around
for debugging, but formal experiments must only run the shared pipeline. This
test is the boundary: no canonical module may import or execute legacy code, and
the legacy assets live under ``legacy/``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
CANONICAL_DIRS = ("dp_manip", "scripts", "slurm", "tests")

# Imports and executions of legacy code, not provenance mentions such as
# ``Ported from VariDP/dp/backbones.py`` in the canonical backbone docstrings.
FORBIDDEN_PATTERNS = (
    re.compile(r"^\s*(?:import|from)\s+varidp\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(?:import|from)\s+dp\b", re.MULTILINE),
    re.compile(r"\btrain_local\b"),
    re.compile(r"\bdp\.dp_lib\b"),
    re.compile(r"\brun_cpu\b"),
    re.compile(r"sys\.path[^\n]*VariDP"),
)

LEGACY_ASSETS = (
    "legacy/README.md",
    "legacy/state/run_cpu.py",
    "legacy/state/mplib-probe-overrides.txt",
    "legacy/state/environment/ubuntu-expert-freeze.txt",
    "legacy/state/manifests/ubuntu-demos.sha256",
    "legacy/state/manifests/wsl-data.sha256",
    "legacy/state/patches/mani_skill_mplib_0_2_1.patch",
    "legacy/state/docs/0923-2016.md",
    "legacy/state/docs/0923-2155.md",
    "legacy/state/docs/0923-2249.md",
    "legacy/state/docs/history.md",
    "legacy/state/docs/instructions.md",
)

MOVED_FROM_ROOT = (
    "run_cpu.py",
    "mplib-probe-overrides.txt",
    "environment",
    "manifests",
    "patches",
    "docs/0923-2016.md",
    "docs/0923-2155.md",
    "docs/0923-2249.md",
    "docs/history.md",
    "docs/instructions.md",
)


class LegacyBoundaryTest(unittest.TestCase):
    def test_canonical_sources_do_not_import_or_execute_legacy_code(self) -> None:
        for directory in CANONICAL_DIRS:
            for path in sorted((ROOT / directory).rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.resolve() == SELF:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in FORBIDDEN_PATTERNS:
                    with self.subTest(path=str(path.relative_to(ROOT)), pattern=pattern.pattern):
                        self.assertIsNone(pattern.search(text))

    def test_legacy_assets_are_staged_under_legacy(self) -> None:
        for relative in LEGACY_ASSETS:
            self.assertTrue((ROOT / relative).is_file(), relative)
        for relative in MOVED_FROM_ROOT:
            self.assertFalse((ROOT / relative).exists(), f"still at the root: {relative}")

    def test_varidp_is_marked_as_a_frozen_donor(self) -> None:
        # The donor itself is kept in the course repository, which does not have to sit
        # next to this checkout (standalone upstream). The archive has to declare it.
        notice = ROOT / "legacy" / "README.md"
        self.assertTrue(notice.is_file())
        text = notice.read_text(encoding="utf-8")
        self.assertIn("VariDP", text)
        self.assertIn("donor", text)

    def test_archived_doc_links_resolve(self) -> None:
        # Moving the archive must not leave dangling relative links.
        pattern = re.compile(r"\]\((?!https?://|mailto:)([^)#]+)\)")
        for path in sorted((ROOT / "legacy").rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for target in pattern.findall(text):
                resolved = (path.parent / target).resolve()
                with self.subTest(path=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(resolved.exists(), f"{target} -> {resolved}")

    def test_entry_points_stay_on_the_shared_pipeline(self) -> None:
        for relative in ("scripts/run_experiment.py", "scripts/train_dp.py", "scripts/sweep.py"):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("dp_manip", source)
                self.assertNotIn("VariDP", source)


if __name__ == "__main__":
    unittest.main()
