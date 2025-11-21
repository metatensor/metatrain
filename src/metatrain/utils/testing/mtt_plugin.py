"""Pytest plugin to add documentation links on test failures.

This plugin is used by architecture tests as ``pytest -p mtt_plugin``
in ``tox.ini``.
"""

from collections.abc import Generator
from typing import Any

import pytest


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
        doc_url = f"https://https://docs.metatensor.org/metatrain/latest/dev-docs/utils/testing.html#{import_path}"

        longrepr = report.longrepr

        # Only modify real traceback objects
        if hasattr(longrepr, "reprtraceback"):
            tb = longrepr.reprtraceback

            message = (
                f"\nCheckout this test's documentation to understand it: \n{doc_url}\n"
            )

            # Add our link *inside* the traceback display
            if tb.extraline:
                tb.extraline += f"\n{message}"
            else:
                tb.extraline = message
