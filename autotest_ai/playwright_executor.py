import json
from playwright.sync_api import sync_playwright, expect


class PlaywrightExecutor:
    def __init__(self, headless: bool = True):
        self.headless = headless

    def execute_test(self, test_steps: list, step_details: list = None) -> dict:
        results = {
            "status": "PASS",
            "steps_executed": 0,
            "error": None,
            "steps": step_details or []
        }

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()

            try:
                for i, step in enumerate(test_steps):
                    try:
                        self._execute_step(page, step)
                        results["steps_executed"] += 1
                        if results["steps"] and i < len(results["steps"]):
                            results["steps"][i]["status"] = "PASS"
                    except Exception as e:
                        results["status"] = "FAIL"
                        error_msg = str(e)
                        results["error"] = error_msg
                        if results["steps"] and i < len(results["steps"]):
                            results["steps"][i]["status"] = "FAIL"
                            results["steps"][i]["error"] = error_msg
                        # Continue running remaining steps
                        results["steps_executed"] += 1

            finally:
                browser.close()

        return results

    def _execute_step(self, page, step: dict):
        action = step.get("action")
        target = step.get("target")
        value = step.get("value")
        assertions = step.get("assertions", [])

        if action == "open":
            page.goto(target, wait_until="domcontentloaded", timeout=15000)

        elif action == "fill":
            if value:
                page.fill(target, value)

        elif action == "click":
            page.click(target, timeout=15000)

        elif action == "wait":
            page.wait_for_timeout(1000)

        for assertion in assertions:
            self._run_assertion(page, assertion)

    def _run_assertion(self, page, assertion: str):
        """Safely evaluate a Playwright assertion string."""
        import re
        try:
            # Build a safe local scope with expect, page, and re
            local_scope = {"expect": expect, "page": page, "re": re}
            eval(assertion, {"__builtins__": {}}, local_scope)
        except Exception as e:
            raise AssertionError(f"Assertion failed [{assertion}]: {str(e)}")


if __name__ == "__main__":
    with open("llm_generated_steps.json", "r") as f:
        test_steps = json.load(f)

    executor = PlaywrightExecutor(headless=True)
    result = executor.execute_test(test_steps)
    print(result)