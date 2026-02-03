"""Test runners for OmniDocs."""

from .registry import Backend, Task, TestSpec, get_test_by_name, get_tests, list_tests
from .report import TestReport, TestSummary, compare_reports, load_report

__all__ = [
    # Registry
    "Backend",
    "Task",
    "TestSpec",
    "get_tests",
    "get_test_by_name",
    "list_tests",
    # Report
    "TestReport",
    "TestSummary",
    "load_report",
    "compare_reports",
]
