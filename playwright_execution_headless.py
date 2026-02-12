from playwright.sync_api import sync_playwright, expect
import re
from typing import List, Dict, Any


class BrowserTestRunner:
    """Runs browser steps and checks assertions."""

    def __init__(self, show_browser: bool = False):
        self.show_browser = show_browser

    def run(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        report = {
            "result": "PASS",
            "completed_steps": 0,
            "message": None
        }

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not self.show_browser)
            page = browser.new_page()

            try:
                for step in steps:
                    self._perform_action(page, step)
                    report["completed_steps"] += 1
            except Exception as err:
                report["result"] = "FAIL"
                report["message"] = str(err)
            finally:
                browser.close()

        return report

    def _perform_action(self, page, step: Dict[str, Any]):
        action_type = step.get("action")
        selector = step.get("target")
        input_value = step.get("value")
        checks = step.get("assertions", [])

        # Perform browser action
        if action_type == "open":
            page.goto(selector)

        elif action_type == "click":
            page.wait_for_selector(selector)
            page.click(selector)

        elif action_type == "fill":
            page.fill(selector, input_value)

        elif action_type == "wait":
            page.wait_for_timeout(int(input_value))

        # Run assertions
        for check in checks:
            if check.strip().startswith("expect("):
                eval(check, {"page": page, "expect": expect, "re": re})


if __name__ == "__main__":
    runner = BrowserTestRunner(show_browser=True)

    demo_steps = [
        {
            "action": "open",
            "target": "https://practice.automationtesting.in/",
            "assertions": [
                "expect(page).to_have_title(re.compile('Automation Practice Site'))"
            ]
        },
        {
            "action": "click",
            "target": "#menu-item-40",
            "assertions": [
                "expect(page).to_have_url(re.compile('.*shop.*'))"
            ]
        }
    ]

    result = runner.run(demo_steps)
    print(result)
