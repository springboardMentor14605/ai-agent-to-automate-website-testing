from typing import Dict, List


class AssertionGenerator:
    """
    Rule-based Assertion Generator for Playwright.
    Converts natural language expectations into Playwright assertions.
    """

    def __init__(self):
        self.visibility_keywords = ["visible", "displayed", "shown"]
        self.hidden_keywords = ["hidden", "not visible", "disappeared"]
        self.text_keywords = ["text", "message", "label"]
        self.url_keywords = ["url", "redirected"]
        self.title_keywords = ["title"]
        self.value_keywords = ["value", "filled", "input"]
        self.enabled_keywords = ["enabled", "clickable"]
        self.disabled_keywords = ["disabled", "not enabled", "not clickable"]

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        assertions = []

        expected = parsed_step.get("expected")
        target = parsed_step.get("target")

        if not expected or not target:
            return assertions

        expected_lower = expected.lower()

        # Visible
        if any(word in expected_lower for word in self.visibility_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_visible()"
            )

        # Hidden
        elif any(word in expected_lower for word in self.hidden_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_hidden()"
            )

        # Enabled
        elif any(word in expected_lower for word in self.enabled_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_enabled()"
            )

        # Disabled
        elif any(word in expected_lower for word in self.disabled_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_disabled()"
            )

        # Text
        elif any(word in expected_lower for word in self.text_keywords):
            expected_text = expected.split("text")[-1].strip().strip("'\"")
            assertions.append(
                f"expect(page.locator('{target}')).to_have_text('{expected_text}')"
            )

        # URL
        elif any(word in expected_lower for word in self.url_keywords):
            expected_url = expected.split()[-1]
            assertions.append(
                f"expect(page).to_have_url('{expected_url}')"
            )

        # Title
        elif any(word in expected_lower for word in self.title_keywords):
            expected_title = expected.split("title")[-1].strip().strip("'\"")
            assertions.append(
                f"expect(page).to_have_title('{expected_title}')"
            )

        # Input value
        elif any(word in expected_lower for word in self.value_keywords):
            expected_value = expected.split("value")[-1].strip().strip("'\"")
            assertions.append(
                f"expect(page.locator('{target}')).to_have_value('{expected_value}')"
            )

        return assertions


# EXECUTION
if __name__ == "__main__":
    generator = AssertionGenerator()


    test_cases = [
    {
        "target": "#submit",
        "expected": "button should be visible"
    },
    {
        "target": "#error",
        "expected": "error message should be hidden"
    },
    {
        "target": "page",
        "expected": "page title Test Login | Practice Test Automation"
    },
    {
        "target": "page",
        "expected": "url should be https://practicetestautomation.com/practice-test-login/"
    },
    {
        "target": "#username",
        "expected": "input value student"
    },
    {
        "target": "#submit",
        "expected": "button should be enabled"
    }
]

    for index, step in enumerate(test_cases, start=1):
        print(f"\nTest Case {index}")
        print("Input:", step)

        assertions = generator.generate_assertions(step)

        if assertions:
            for assertion in assertions:
                print("Generated Assertion:", assertion)
        else:
            print("No assertion generated")