import json
from playwright.sync_api import sync_playwright, expect


class PlaywrightExecutor:
    def __init__(self, headless: bool = True):
        self.headless = headless

    def execute_test(self, test_steps: list) -> dict:
        results = {
            "status": "PASS",
            "steps_executed": 0,
            "error": None
        }

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()

            try:
                for step in test_steps:
                    self._execute_step(page, step)
                    results["steps_executed"] += 1

            except Exception as e:
                results["status"] = "FAIL"
                results["error"] = str(e)

            finally:
                browser.close()

        return results

    def _execute_step(self, page, step: dict):
        action = step.get("action")
        target = step.get("target")
        value = step.get("value")
        assertions = step.get("assertions", [])

        if action == "open":
            page.goto(target)

        elif action == "fill":
            page.fill(target, value)

        elif action == "click":
            page.click(target)

        elif action == "wait":
            # page.wait_for_timeout(int(value))
            pass

        for assertion in assertions:
            eval(assertion)


if __name__ == "__main__":
    # Load LLM-generated assertions
    with open("llm_generated_steps.json", "r") as f:
        test_steps = json.load(f)

    executor = PlaywrightExecutor(headless=True)
    result = executor.execute_test(test_steps)
    print(result)