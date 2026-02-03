"""
Test Result Reporting for OmniDocs.

Provides utilities for formatting, aggregating, and exporting test results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestSummary:
    """Summary statistics for a test run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total_time: float = 0.0
    avg_load_time: float = 0.0
    avg_inference_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Percentage of tests that passed."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


@dataclass
class TestReport:
    """Complete test report with results and metadata."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: TestSummary = field(default_factory=TestSummary)
    results: List[Dict[str, Any]] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a test result to the report."""
        self.results.append(result)
        self.summary.total += 1

        if result.get("success"):
            self.summary.passed += 1
        elif result.get("skipped"):
            self.summary.skipped += 1
        else:
            self.summary.failed += 1

        load_time = result.get("load_time", 0)
        inference_time = result.get("inference_time", 0)
        self.summary.total_time += load_time + inference_time

    def finalize(self) -> None:
        """Calculate final statistics."""
        if self.summary.total > 0:
            total_load = sum(r.get("load_time", 0) for r in self.results)
            total_inference = sum(r.get("inference_time", 0) for r in self.results)
            self.summary.avg_load_time = total_load / self.summary.total
            self.summary.avg_inference_time = total_inference / self.summary.total

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.summary.total,
                "passed": self.summary.passed,
                "failed": self.summary.failed,
                "skipped": self.summary.skipped,
                "pass_rate": self.summary.pass_rate,
                "total_time": self.summary.total_time,
                "avg_load_time": self.summary.avg_load_time,
                "avg_inference_time": self.summary.avg_inference_time,
            },
            "filters": self.filters,
            "results": self.results,
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Export report to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            path.write_text(json_str)
        return json_str

    def to_markdown(self, path: Optional[Path] = None) -> str:
        """Export report to Markdown."""
        lines = [
            "# OmniDocs Test Report",
            "",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            "",
            f"- **Total Tests:** {self.summary.total}",
            f"- **Passed:** {self.summary.passed}",
            f"- **Failed:** {self.summary.failed}",
            f"- **Skipped:** {self.summary.skipped}",
            f"- **Pass Rate:** {self.summary.pass_rate:.1f}%",
            f"- **Total Time:** {self.summary.total_time:.2f}s",
            f"- **Avg Load Time:** {self.summary.avg_load_time:.2f}s",
            f"- **Avg Inference Time:** {self.summary.avg_inference_time:.2f}s",
            "",
        ]

        if self.filters:
            lines.extend([
                "## Filters Applied",
                "",
            ])
            for key, value in self.filters.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Results table
        lines.extend([
            "## Results",
            "",
            "| Test | Backend | Task | Status | Load (s) | Inference (s) |",
            "|------|---------|------|--------|----------|---------------|",
        ])

        for r in self.results:
            status = "PASS" if r.get("success") else "FAIL"
            status_emoji = "✓" if r.get("success") else "✗"
            lines.append(
                f"| {r.get('test_name', 'Unknown')} | "
                f"{r.get('backend', '-')} | "
                f"{r.get('task', '-')} | "
                f"{status_emoji} {status} | "
                f"{r.get('load_time', 0):.2f} | "
                f"{r.get('inference_time', 0):.2f} |"
            )

        # Failed tests details
        failed = [r for r in self.results if not r.get("success")]
        if failed:
            lines.extend([
                "",
                "## Failed Tests",
                "",
            ])
            for r in failed:
                lines.extend([
                    f"### {r.get('test_name', 'Unknown')}",
                    "",
                    f"**Error:** {r.get('error', 'Unknown error')}",
                    "",
                ])

        md_str = "\n".join(lines)
        if path:
            path.write_text(md_str)
        return md_str

    def print_summary(self) -> None:
        """Print a summary to stdout."""
        print("\n" + "=" * 75)
        print("TEST SUMMARY")
        print("=" * 75)
        print(f"Total:     {self.summary.total}")
        print(f"Passed:    {self.summary.passed}")
        print(f"Failed:    {self.summary.failed}")
        print(f"Skipped:   {self.summary.skipped}")
        print(f"Pass Rate: {self.summary.pass_rate:.1f}%")
        print(f"Time:      {self.summary.total_time:.2f}s")
        print("=" * 75)

        if self.summary.failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.get("success"):
                    print(f"  - {r.get('test_name')}: {r.get('error', 'Unknown')}")


def load_report(path: Path) -> TestReport:
    """Load a report from a JSON file."""
    data = json.loads(path.read_text())

    report = TestReport(
        timestamp=data.get("timestamp", ""),
        filters=data.get("filters", {}),
    )

    # Reconstruct summary
    summary_data = data.get("summary", {})
    report.summary = TestSummary(
        total=summary_data.get("total", 0),
        passed=summary_data.get("passed", 0),
        failed=summary_data.get("failed", 0),
        skipped=summary_data.get("skipped", 0),
        total_time=summary_data.get("total_time", 0.0),
        avg_load_time=summary_data.get("avg_load_time", 0.0),
        avg_inference_time=summary_data.get("avg_inference_time", 0.0),
    )

    report.results = data.get("results", [])
    return report


def compare_reports(
    baseline: TestReport,
    current: TestReport,
) -> Dict[str, Any]:
    """Compare two test reports and return differences."""
    baseline_tests = {r["test_name"]: r for r in baseline.results}
    current_tests = {r["test_name"]: r for r in current.results}

    regressions = []
    improvements = []
    new_tests = []
    removed_tests = []

    for name, result in current_tests.items():
        if name not in baseline_tests:
            new_tests.append(name)
            continue

        baseline_result = baseline_tests[name]

        # Check for regressions (was passing, now failing)
        if baseline_result.get("success") and not result.get("success"):
            regressions.append({
                "test": name,
                "error": result.get("error"),
            })
        # Check for improvements (was failing, now passing)
        elif not baseline_result.get("success") and result.get("success"):
            improvements.append(name)

    for name in baseline_tests:
        if name not in current_tests:
            removed_tests.append(name)

    return {
        "regressions": regressions,
        "improvements": improvements,
        "new_tests": new_tests,
        "removed_tests": removed_tests,
        "baseline_pass_rate": baseline.summary.pass_rate,
        "current_pass_rate": current.summary.pass_rate,
        "pass_rate_delta": current.summary.pass_rate - baseline.summary.pass_rate,
    }
