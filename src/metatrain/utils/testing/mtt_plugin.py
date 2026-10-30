"""Pytest plugin for architecture tests.

This plugin adds documentation links on test failures, and applies the per-test
timeouts declared in the ``timeouts`` attribute of test classes. It is loaded
with ``-p mtt_plugin`` from ``pyproject.toml``.
"""

from collections.abc import Generator
from typing import Any

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """
    Add a timeout marker to tests listed in their class ``timeouts`` attribute.

    :param items: The collected test items.
    """
    checked: set[type] = set()
    for item in items:
        cls = getattr(item, "cls", None)
        timeouts = getattr(cls, "timeouts", None)
        if cls is None or not timeouts:
            continue

        if cls not in checked:
            checked.add(cls)
            for name in timeouts:
                if not callable(getattr(cls, name, None)):
                    raise pytest.UsageError(
                        f"{cls.__qualname__}.timeouts refers to '{name}', "
                        "which is not a test in this class"
                    )

        timeout = timeouts.get(getattr(item, "originalname", item.name))
        if timeout is not None:
            # prepend the marker so it takes precedence over any decorator
            item.add_marker(pytest.mark.timeout(timeout), append=False)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator:
    """
    Add a documentation link to the report whenever a test fails.

    :param item: The pytest item object.
    :param call: The pytest call object.

    :return: A generator
    """
    outcome: Any = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Build link dynamically based on test name
        module = item.obj.__module__
        if not module.startswith("metatrain.utils.testing"):
            return

        import_path = f"metatrain.utils.testing.{item.obj.__qualname__}"

        # Example: customize your documentation URL here
        doc_url = f"https://docs.metatensor.org/metatrain/latest/dev-docs/utils/testing.html#{import_path}"

        longrepr = report.longrepr

        # Only modify real traceback objects
        if hasattr(longrepr, "reprtraceback"):
            tb = longrepr.reprtraceback

            message = (
                f"\nCheck out this test's documentation to understand it: \n{doc_url}\n"
            )

            # Add our link *inside* the traceback display
            if tb.extraline:
                tb.extraline += f"\n{message}"
            else:
                tb.extraline = message
