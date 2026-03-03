import pytest
import os

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    
    # Set a report attribute for each phase of a call (setup, call, teardown)
    setattr(item, "rep_" + rep.when, rep)
