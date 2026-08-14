"""
TEST-010 — Compile the test report.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Runs the suite, parses pytest's own JUnit XML and coverage XML, and renders
``tests/test_report.md``.

Acceptance (TEST-010): the report is generated from actual pytest output, not
summarized from memory. That is why this is a script and not a hand-written
document — the numbers in the report are read out of the files pytest wrote
seconds earlier, and regenerating after a change cannot produce a stale figure.

Usage::

    python -m scripts.generate_test_report              # full suite + coverage
    python -m scripts.generate_test_report --no-cov     # faster, no coverage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "tests" / "test_report.md"
ARTIFACT_DIR = REPO_ROOT / "benchmark" / "results"
JUNIT_PATH = ARTIFACT_DIR / "pytest_junit.xml"
COVERAGE_PATH = ARTIFACT_DIR / "coverage.xml"

#: Which suite covers which owner's work, for the per-milestone breakdown.
SUITE_OWNERS = {
    "tests/common": "Shared (Member 1 — DS-005 normalization)",
    "tests/member2": "Member 2 — annotation, validation, gold set",
    "tests/member3": "Member 3 — embeddings, ChromaDB, retrieval",
    "tests/member4": "Member 4 — RAG pipeline, evaluation engine",
    "tests/member5": "Member 5 — FastAPI, integration contract",
    "tests/member6": "Member 6 — benchmarking, testing, deployment",
    "tests/test_api.py": "Member 5 — API integration (API-014)",
    "tests/test_downstream_mock.py": "Member 5 — downstream mocks (INT-008)",
}

#: Member 6's TEST-00N tasks and the module that discharges each.
TASK_MODULES = {
    "TEST-001": "tests/member6/test_dataset_validation.py",
    "TEST-002": "tests/member6/test_embedding.py",
    "TEST-003": "tests/member6/test_vectorstore.py",
    "TEST-004": "tests/member6/test_rag_context.py",
    "TEST-005": "tests/member6/test_scoring_and_bands.py",
    "TEST-006": "tests/member6/test_llm_mock.py",
    "TEST-007": "tests/member6/test_e2e.py",
    "TEST-008": "tests/member6/test_regression.py",
    "TEST-009": "tests/member6/test_malformed_input.py",
}


def run_suite(with_coverage: bool = True) -> Dict[str, Any]:
    """Run pytest, writing the JUnit (and optionally coverage) XML."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable, "-m", "pytest",
        "-q", "--no-header", "-p", "no:cacheprovider",
        f"--junit-xml={JUNIT_PATH}",
    ]
    if with_coverage:
        args += [
            "--cov=src",
            "--cov=benchmark",
            "--cov-report", f"xml:{COVERAGE_PATH}",
            "--cov-report", "term-missing:skip-covered",
        ]

    completed = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": " ".join(str(part) for part in args),
    }


def _module_path(case: ET.Element) -> str:
    """Resolve a testcase to its repo-relative module path.

    pytest's JUnit output carries a dotted ``classname``
    (``tests.member6.test_e2e``) and, depending on version, no ``file``
    attribute at all — so the path has to be reconstructed. For class-based
    tests the last dotted segment is the class, not a module, hence the retry.
    """
    explicit = case.get("file")
    if explicit:
        return explicit.replace("\\", "/")

    dotted = case.get("classname", "")
    if not dotted:
        return "unknown"

    parts = dotted.split(".")
    for cut in (len(parts), len(parts) - 1):
        if cut <= 0:
            break
        candidate = "/".join(parts[:cut]) + ".py"
        if (REPO_ROOT / candidate).exists():
            return candidate

    return "/".join(parts) + ".py"


def parse_junit(path: Path = JUNIT_PATH) -> Dict[str, Any]:
    """Extract per-file results from pytest's JUnit XML."""
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root

    totals = {
        "tests": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "time": float(suite.get("time", 0.0)),
    }
    totals["passed"] = (
        totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    )

    per_file: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    )
    problems: List[Dict[str, str]] = []

    for case in suite.iter("testcase"):
        filename = _module_path(case)
        entry = per_file[filename]
        entry["tests"] += 1
        entry["time"] += float(case.get("time", 0.0))

        if case.find("failure") is not None:
            entry["failures"] += 1
            problems.append(
                {
                    "file": filename,
                    "name": case.get("name", ""),
                    "kind": "failure",
                    "message": (case.find("failure").get("message") or "")[:300],
                }
            )
        elif case.find("error") is not None:
            entry["errors"] += 1
            problems.append(
                {
                    "file": filename,
                    "name": case.get("name", ""),
                    "kind": "error",
                    "message": (case.find("error").get("message") or "")[:300],
                }
            )
        elif case.find("skipped") is not None:
            entry["skipped"] += 1
            problems.append(
                {
                    "file": filename,
                    "name": case.get("name", ""),
                    "kind": "skipped",
                    "message": (case.find("skipped").get("message") or "")[:300],
                }
            )

    return {"totals": totals, "per_file": dict(per_file), "problems": problems}


def parse_coverage(path: Path = COVERAGE_PATH) -> Optional[Dict[str, Any]]:
    """Extract per-package line coverage from coverage.xml."""
    if not path.exists():
        return None

    root = ET.parse(path).getroot()
    overall = round(float(root.get("line-rate", 0.0)) * 100, 1)

    packages: Dict[str, Dict[str, Any]] = {}
    for package in root.iter("package"):
        name = package.get("name", "")
        packages[name] = {
            "line_rate": round(float(package.get("line-rate", 0.0)) * 100, 1),
            "files": len(list(package.iter("class"))),
        }

    weakest = sorted(
        (
            {
                "file": klass.get("filename", ""),
                "line_rate": round(float(klass.get("line-rate", 0.0)) * 100, 1),
            }
            for klass in root.iter("class")
        ),
        key=lambda entry: entry["line_rate"],
    )[:10]

    return {"overall": overall, "packages": packages, "weakest": weakest}


def _suite_for(filename: str) -> str:
    for prefix, owner in SUITE_OWNERS.items():
        if filename.startswith(prefix):
            return prefix
    return "other"


def render(
    junit: Dict[str, Any],
    coverage: Optional[Dict[str, Any]],
    run: Dict[str, Any],
) -> str:
    totals = junit["totals"]
    per_file = junit["per_file"]

    by_suite: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    )
    for filename, entry in per_file.items():
        bucket = by_suite[_suite_for(filename)]
        for key in ("tests", "failures", "errors", "skipped"):
            bucket[key] += entry[key]

    status = "PASSING" if totals["failures"] == 0 and totals["errors"] == 0 else "FAILING"

    lines: List[str] = [
        "# PECS Evaluation API — Test Report",
        "",
        "Task: **TEST-010** · Owner: Member 6 (Testing + Benchmarking + Docker + "
        "Deployment)",
        "",
        "> Generated by `python -m scripts.generate_test_report`. Every number "
        "below is parsed from the JUnit and coverage XML pytest wrote during "
        "that run (`benchmark/results/pytest_junit.xml`, "
        "`benchmark/results/coverage.xml`). Nothing here is transcribed by hand.",
        "",
        "## 1. Summary",
        "",
        f"**Status: {status}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tests collected | {totals['tests']} |",
        f"| Passed | {totals['passed']} |",
        f"| Failed | {totals['failures']} |",
        f"| Errors | {totals['errors']} |",
        f"| Skipped | {totals['skipped']} |",
        f"| Wall clock | {totals['time']:.1f}s |",
        f"| Exit code | {run['returncode']} |",
        "",
        f"Command: `{run['command'].replace(str(REPO_ROOT), '.')}`",
        "",
        "## 2. By suite",
        "",
        "| Suite | Owner | Tests | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]

    for prefix in sorted(by_suite):
        entry = by_suite[prefix]
        lines.append(
            f"| `{prefix}` | {SUITE_OWNERS.get(prefix, '—')} | {entry['tests']} | "
            f"{entry['failures'] + entry['errors']} | {entry['skipped']} |"
        )

    lines += [
        "",
        "## 3. Member 6 task coverage (TEST-001 … TEST-009)",
        "",
        "| Task | Module | Tests | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]

    for task, module in TASK_MODULES.items():
        entry = per_file.get(module)
        if entry is None:
            lines.append(f"| {task} | `{module}` | **module not found** | — | — |")
            continue
        lines.append(
            f"| {task} | `{module}` | {entry['tests']} | "
            f"{entry['failures'] + entry['errors']} | {entry['skipped']} |"
        )

    lines += ["", "## 4. Coverage", ""]

    if coverage is None:
        lines += [
            "Coverage was not measured in this run (`--no-cov`). Re-run with:",
            "",
            "```bash",
            "pip install -r requirements-dev.txt",
            "python -m scripts.generate_test_report",
            "```",
            "",
        ]
    else:
        lines += [
            f"Overall line coverage of `src/` and `benchmark/`: "
            f"**{coverage['overall']}%**",
            "",
            "| Package | Line coverage | Files |",
            "|---|---|---|",
        ]
        for name in sorted(coverage["packages"]):
            entry = coverage["packages"][name]
            lines.append(f"| `{name}` | {entry['line_rate']}% | {entry['files']} |")

        lines += [
            "",
            "Least-covered modules — the honest list of where a regression would "
            "currently go unnoticed:",
            "",
            "| Module | Line coverage |",
            "|---|---|",
        ]
        for entry in coverage["weakest"]:
            lines.append(f"| `{entry['file']}` | {entry['line_rate']}% |")
        lines.append("")

    problems = junit["problems"]
    failures = [entry for entry in problems if entry["kind"] in ("failure", "error")]
    skipped = [entry for entry in problems if entry["kind"] == "skipped"]

    lines += ["## 5. Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| Module | Test | Message |", "|---|---|---|"]
        for entry in failures:
            message = entry["message"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{entry['file']}` | {entry['name']} | {message} |")
    lines.append("")

    lines += ["## 6. Skipped", ""]
    if not skipped:
        lines.append("None.")
    else:
        lines += [
            "Skips are environment-conditional, not disabled tests — each one "
            "names the dependency it needs.",
            "",
            "| Module | Test | Reason |",
            "|---|---|---|",
        ]
        for entry in skipped:
            message = entry["message"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{entry['file']}` | {entry['name']} | {message} |")
    lines.append("")

    lines += [
        "## 7. Known gaps",
        "",
        "What this suite does **not** cover, stated so nobody infers it from a "
        "green run:",
        "",
        "- **No live-model tests.** Every LLM interaction is a scripted fake "
        "(TEST-006's acceptance criterion). Grading quality against a real model "
        "is BENCH-004/BENCH-010's job and requires `ANTHROPIC_API_KEY`.",
        "- **No container tests.** The Dockerfile and compose invariants are "
        "checked statically (`tests/member6/test_deployment.py`); the build, "
        "health, persistence and secret checks need a Docker engine and are run "
        "by `python -m scripts.verify_container`.",
        "- **No load or concurrency testing.** The rate limiter is unit-tested; "
        "behaviour under sustained concurrent traffic is unmeasured.",
        "- **Workstream B modules are absent**, so TEST-001 covers the "
        "annotation-layer and vector-store-layer validation that exists rather "
        "than Member 1's `schema.py` / `quarantine.py`. See "
        "`docs/known_issues_and_limitations.md`.",
        "",
        "## 8. Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt -r requirements-dev.txt",
        "python -m scripts.generate_test_report",
        "```",
        "",
    ]

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TEST-010 test report generator")
    parser.add_argument("--no-cov", action="store_true", help="skip coverage measurement")
    parser.add_argument("--skip-run", action="store_true", help="reuse existing XML")
    args = parser.parse_args(argv)

    if args.skip_run:
        if not JUNIT_PATH.exists():
            print(f"no JUnit XML at {JUNIT_PATH}; run without --skip-run", file=sys.stderr)
            return 1
        run = {"returncode": -1, "command": "(reused existing artifacts)", "stdout": "", "stderr": ""}
    else:
        run = run_suite(with_coverage=not args.no_cov)
        print(run["stdout"][-3000:])

    junit = parse_junit()
    coverage = None if args.no_cov else parse_coverage()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(junit, coverage, run), encoding="utf-8")

    print(f"wrote {REPORT_PATH}")
    print(json.dumps(junit["totals"], indent=2))

    return 0 if junit["totals"]["failures"] == 0 and junit["totals"]["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main(sys.argv[1:]))
