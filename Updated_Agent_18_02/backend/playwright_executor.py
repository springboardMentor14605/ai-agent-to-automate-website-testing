"""
Playwright Executor - Subprocess-based approach.

Runs Playwright in a completely separate Python process to avoid
event loop conflicts with FastAPI/Uvicorn on Windows.

Includes post-login verification: checks if the login actually
succeeded by detecting URL changes and on-page error messages.
"""
import os
import sys
import json
import datetime
import subprocess
import tempfile


class PlaywrightExecutor:
    """
    Executes Playwright tests by spawning a separate subprocess.
    This avoids all asyncio event loop conflicts on Windows.
    """
    def __init__(self, headless: bool = True, slow_mo: int = 0, timeout: int = 30000):
        self.headless = headless
        self.slow_mo = slow_mo
        self.default_timeout = timeout

    def execute_test(self, test_steps: list, login_url: str = "") -> dict:
        """
        Execute test steps by writing them to a temp file and running
        a subprocess with the sync Playwright API.

        Args:
            test_steps: List of step dicts with action/target/value.
            login_url:  The original login URL to compare against after login.
        """
        results = {
            "status": "PASS",
            "steps_executed": 0,
            "error": None,
            "screenshot": None,
            "details": []
        }

        # Create screenshots directory
        screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshots_dir, f"result_{timestamp}.png")

        # Build and write the runner script to a temp file
        runner_code = self._build_runner_script(test_steps, screenshot_path, login_url)

        tmp_script = None
        try:
            tmp_script = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False,
                encoding='utf-8'
            )
            tmp_script.write(runner_code)
            tmp_script.close()

            python_exe = sys.executable
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.run(
                [python_exe, tmp_script.name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.default_timeout // 1000 + 30,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode == 0 and stdout:
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            results = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    results["status"] = "FAIL"
                    results["error"] = "Could not parse runner output"
                    results["details"].append(f"stdout: {stdout[-500:]}")
            else:
                results["status"] = "FAIL"
                error_msg = stderr if stderr else stdout if stdout else "Unknown subprocess error"
                results["error"] = error_msg[-1000:]
                results["details"].append("[FAIL] Subprocess failed")

        except subprocess.TimeoutExpired:
            results["status"] = "FAIL"
            results["error"] = f"Test timed out after {self.default_timeout // 1000 + 30} seconds"
            results["details"].append("[FAIL] Timeout")
        except Exception as e:
            results["status"] = "FAIL"
            results["error"] = f"{type(e).__name__}: {str(e)}"
            results["details"].append(f"[FAIL] Error: {str(e)}")
        finally:
            if tmp_script and os.path.exists(tmp_script.name):
                try:
                    os.unlink(tmp_script.name)
                except Exception:
                    pass

        return results

    def _build_runner_script(self, test_steps: list, screenshot_path: str, login_url: str) -> str:
        """Build a standalone Python script that runs the Playwright test."""
        steps_json = json.dumps(test_steps)
        script = f'''# -*- coding: utf-8 -*-
import json
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Common selectors for error messages on login pages
ERROR_SELECTORS = [
    "[data-test='error']",
    ".error-message-container",
    ".error-message",
    ".error",
    ".alert-danger",
    ".alert-error",
    ".login-error",
    ".form-error",
    ".notification-error",
    ".flash-error",
    "#flash-messages .error",
    ".flash.error",
    "[role='alert']",
    ".invalid-feedback:visible",
    ".field-error",
    ".auth-error",
]

def find_error_on_page(page):
    """
    Scan the page for visible error messages after login attempt.
    Returns the error text if found, or None.
    """
    for selector in ERROR_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                text = el.inner_text().strip()
                if text:
                    return text
        except Exception:
            continue
    return None

def run_test():
    from playwright.sync_api import sync_playwright

    test_steps = json.loads({repr(steps_json)})
    screenshot_path = {repr(screenshot_path)}
    original_login_url = {repr(login_url)}

    results = {{
        "status": "PASS",
        "steps_executed": 0,
        "error": None,
        "screenshot": None,
        "details": []
    }}

    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless={str(self.headless)}, slow_mo={self.slow_mo})
            context = browser.new_context(viewport={{"width": 1280, "height": 720}})
            page = context.new_page()
            page.set_default_timeout({self.default_timeout})

            url_before_login = ""

            def smart_locator(page, selector, timeout=15000):
                """Try multiple strategies to find an element."""
                # Try locator first (works for CSS selectors and text= selectors)
                try:
                    loc = page.locator(selector)
                    if loc.count() > 0:
                        loc.nth(0).wait_for(state="visible", timeout=timeout)
                        return loc.nth(0)
                except Exception:
                    pass

                # If selector looks like a CSS selector, try wait_for_selector
                try:
                    page.wait_for_selector(selector, state="visible", timeout=timeout)
                    loc = page.locator(selector)
                    if loc.count() > 0:
                        return loc.nth(0)
                except Exception:
                    pass

                # Try placeholder match
                try:
                    # extract quoted text if selector contains placeholder='text'
                    text_sel = selector
                    if "placeholder" in selector:
                        for quote in ["'", '"']:
                            if quote in selector:
                                parts = selector.split(quote)
                                if len(parts) >= 2:
                                    text_sel = parts[1]
                                    break
                    loc = page.get_by_placeholder(text_sel, exact=False)
                    if loc.count() > 0:
                        loc.nth(0).wait_for(state="visible", timeout=timeout)
                        return loc.nth(0)
                except Exception:
                    pass

                # Try label match
                try:
                    loc = page.get_by_label(selector, exact=False)
                    if loc.count() > 0:
                        loc.nth(0).wait_for(state="visible", timeout=timeout)
                        return loc.nth(0)
                except Exception:
                    pass

                # Try text match
                try:
                    loc = page.get_by_text(selector, exact=False)
                    if loc.count() > 0:
                        loc.nth(0).wait_for(state="visible", timeout=timeout)
                        return loc.nth(0)
                except Exception:
                    pass

                # Try role-based heuristics
                try:
                    lower = selector.lower()
                    if any(kw in lower for kw in ["submit", "button", "login", "sign", "continue"]):
                        loc = page.get_by_role("button")
                        if loc.count() > 0:
                            loc.nth(0).wait_for(state="visible", timeout=timeout)
                            return loc.nth(0)
                except Exception:
                    pass

                raise Exception(f"Could not find element: {selector}")

            for idx, step in enumerate(test_steps):
                action = step.get("action")
                target = step.get("target")
                value = step.get("value", "")
                step_label = f"Step {{idx + 1}}: {{action}} {{target or ''}}"

                # Guard: empty selector for actions that need one
                if action in ("click", "fill") and not target:
                    raise Exception(f"Empty selector for '{{action}}' action at step {{idx + 1}}. The page scout could not identify a valid CSS selector for this element.")

                try:
                    if action == "open":
                        page.goto(target, wait_until="domcontentloaded")
                        try:
                            page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        # Close any popups/overlays that might appear
                        page.wait_for_timeout(2000)
                        # Try to dismiss common popups
                        for dismiss_sel in ["button[aria-label='Close']", ".close-btn", "[data-dismiss]", "button:has-text('✕')", "button:has-text('Close')", "button:has-text('Not now')"]:
                            try:
                                dismiss = page.locator(dismiss_sel)
                                if dismiss.count() > 0 and dismiss.first.is_visible():
                                    dismiss.first.click(timeout=2000)
                                    page.wait_for_timeout(500)
                            except Exception:
                                pass
                        url_before_login = page.url

                    elif action == "click":
                        loc = smart_locator(page, target)
                        url_before_login = page.url
                        # Attempt click and handle potential new page/window
                        try:
                            with context.expect_page(timeout=3000) as page_info:
                                loc.click()
                            new_page = page_info.value
                            # switch to new page if opened
                            if new_page:
                                page = new_page
                                page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except Exception:
                            # no new page opened, proceed normally
                            try:
                                loc.click()
                            except Exception:
                                raise
                        page.wait_for_timeout(1500)

                    elif action == "fill":
                        loc = smart_locator(page, target)
                        loc.fill(value)
                        page.wait_for_timeout(300)

                    elif action == "select":
                        loc = smart_locator(page, target)
                        loc.select_option(value)
                        page.wait_for_timeout(300)

                    elif action == "scroll":
                        direction = -300 if value == "up" else 300
                        if target:
                            page.locator(target).evaluate(f"el => el.scrollBy(0, {{direction}})")
                        else:
                            page.evaluate(f"window.scrollBy(0, {{direction}})")
                        page.wait_for_timeout(300)

                    elif action == "wait":
                        page.wait_for_timeout(int(value))

                    elif action == "press":
                        page.keyboard.press(value or "Enter")
                        page.wait_for_timeout(500)

                    results["steps_executed"] += 1
                    results["details"].append(f"[PASS] {{step_label}}")

                except Exception as step_err:
                    results["details"].append(f"[FAIL] {{step_label}} - {{str(step_err)}}")
                    raise

            # ============================================
            # POST-LOGIN VERIFICATION
            # ============================================
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            page.wait_for_timeout(500)
            url_after_login = page.url

            # 1. Check for visible error messages on the page
            error_text = find_error_on_page(page)
            if error_text:
                results["status"] = "FAIL"
                results["error"] = f"Login failed: {{error_text}}"
                results["details"].append(f"[FAIL] Login verification: error message detected on page")
                results["details"].append(f"[INFO] Error message: {{error_text}}")

            # 2. Check if URL changed (successful login usually redirects)
            elif url_before_login and url_after_login:
                # Normalize URLs for comparison (strip trailing slashes)
                norm_before = url_before_login.rstrip("/")
                norm_after = url_after_login.rstrip("/")
                if norm_before == norm_after:
                    # URL didn't change — likely login failed silently
                    # Double-check page title or content for clues
                    page_title = page.title().lower()
                    page_text = page.inner_text("body")[:500].lower()

                    login_keywords = ["login", "sign in", "log in", "signin", "authenticate"]
                    still_on_login = any(kw in page_title or kw in page_text[:200] for kw in login_keywords)

                    if still_on_login:
                        results["status"] = "FAIL"
                        results["error"] = "Login failed: page did not navigate away from the login page. Credentials may be incorrect."
                        results["details"].append("[FAIL] Login verification: still on login page after submission")
                    else:
                        results["details"].append("[PASS] Login verification: page content changed")
                else:
                    results["details"].append(f"[PASS] Login verification: redirected to {{url_after_login}}")

            # Take screenshot (always, regardless of pass/fail)
            page.screenshot(path=screenshot_path)
            results["screenshot"] = screenshot_path
            results["details"].append("[PASS] Screenshot captured")

    except Exception as e:
        results["status"] = "FAIL"
        results["error"] = f"{{type(e).__name__}}: {{str(e)}}"
        try:
            if "page" in dir() and page:
                page.screenshot(path=screenshot_path)
                results["screenshot"] = screenshot_path
        except Exception:
            pass
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass

    return results

if __name__ == "__main__":
    result = run_test()
    print(json.dumps(result, ensure_ascii=True))
'''
        return script
