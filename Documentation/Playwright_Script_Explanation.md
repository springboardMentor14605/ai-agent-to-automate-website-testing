# Playwright Test Script Documentation

This document explains the structure and functionality of the automated testing script used for the login page. The setup uses **Python**, **Pytest**, and **Playwright** to perform browser automation and verify application behavior.

## Project Structure

*   **`test_playwright.py`**: Contains the test cases and Playwright fixtures.
*   **`conftest.py`**: Contains Pytest configuration and hooks (specifically for reporting).
*   **`login.html`**: The local web page being tested.
*   **`screenshots/`**: Directory where screenshots of failed tests are stored.

---

## 1. `conftest.py`: The Reporting Hook

This file is critical for determining whether a test passed or failed, which we need to know to take a screenshot.

### `pytest_runtest_makereport`
This is a standard Pytest hook that is triggered when a test report is being created.

```python
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # ...
    setattr(item, "rep_" + rep.when, rep)
```

**How it works:**
1.  **Intercepts Execution**: It wraps the test execution.
2.  **Captures Result**: It gets the result of the test (Pass/Fail/Error).
3.  **Attaches to Item**: It attaches this result (`rep`) to the test item object (`item`). This allows us to inspect `item.rep_call.failed` later inside our tests or fixtures.

---

## 2. `test_playwright.py`: Fixtures and Tests

This file contains the actual test logic.

### Configuration
*   **`BASE_DIR`**: Calculates the absolute path of the script's directory.
*   **`SCREENSHOT_DIR`**: Defines where screenshots are saved (`./screenshots`).
*   **`LOGIN_PAGE`**: Constructs the `file:///` URL for `login.html` dynamically, ensuring it works on any machine.

### Fixtures
Fixtures are setup/teardown helpers that run before/after tests.

1.  **`playwright_instance` (Session Scope)**:
    *   Starts the Playwright process once for the entire test session.
2.  **`browser` (Session Scope)**:
    *   Launches the Chromium browser once for the entire session.
    *   `headless=False` means you can see the browser open.
3.  **`page` (Function Scope)**:
    *   **Setup**: Creates a new browser context and page for *each* test function. This ensures isolation (cookies/storage doesn't leak between tests).
    *   **Teardown & Screenshot Logic**:
        *   After the test finishes (`yield` returns), the teardown code runs.
        *   It checks `request.node.rep_call.failed` (set by `conftest.py`).
        *   **If Failed**: It takes a full-page screenshot and saves it to the `screenshots/` folder with a name like `test_negative_login_failed.png`.

### Tests

#### `test_positive_login(page)`
*   **Goal**: Verify successful login.
*   **Steps**:
    1.  Go to `login.html`.
    2.  Fill in correct username and password.
    3.  Click Login.
    4.  **Assert**: Check if `#successMessage` contains "Login Successful!".
*   **Outcome**: PASS.

#### `test_negative_login(page)`
*   **Goal**: Demonstrate failure handling.
*   **Steps**:
    1.  Go to `login.html`.
    2.  Fill in empty/wrong credentials.
    3.  Click Login.
    4.  **Assert**: Checks for "WRONG EXPECTED TEXT".
*   **Outcome**: FAIL (Intentionally).
    *   Because this fails, the `page` fixture automatically captures a screenshot.

---

## How to Run

Open your terminal in the project directory and run:

```bash
pytest test_playwright.py
```

To see detailed output (pass/fail status):

```bash
pytest -v test_playwright.py
```
