from typing import Dict, List

class AssertionGenerator:
    """
    Generates Playwright assertions based on
    parsed expected outcomes from natural language.
    """
    def __init__(self):
        self.visibility_keywords = ["visible", "displayed", "shown"]
        self.hidden_keywords = ["hidden", "not visible", "disappeared"]
        self.text_keywords = ["text", "message", "label"]
        self.url_keywords = ["url", "redirected"]
        self.title_keywords = ["title"]
        self.value_keywords = ["value", "filled", "input"]

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        assertions = []
        expected = parsed_step.get("expected")
        target = parsed_step.get("target")

        if not expected:
            return assertions

        expected_lower = expected.lower()

        if any(word in expected_lower for word in self.visibility_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_visible()"
            )
        elif any(word in expected_lower for word in self.hidden_keywords):
            assertions.append(
                f"expect(page.locator('{target}')).to_be_hidden()"
            )
        elif any(word in expected_lower for word in self.text_keywords):
            expected_text = self._extract_expected_text(expected)
            assertions.append(
                f"expect(page.locator('{target}')).to_have_text('{expected_text}')"
            )
        elif any(word in expected_lower for word in self.url_keywords):
            expected_url = self._extract_expected_url(expected)
            assertions.append(
                f"expect(page).to_have_url('{expected_url}')"
            )
        elif any(word in expected_lower for word in self.title_keywords):
            expected_title = self._extract_expected_title(expected)
            assertions.append(
                f"expect(page).to_have_title('{expected_title}')"
            )
        elif any(word in expected_lower for word in self.value_keywords):
            expected_value = self._extract_expected_value(expected)
            assertions.append(
                f"expect(page.locator('{target}')).to_have_value('{expected_value}')"
            )
        
        return assertions

    def _extract_expected_text(self, expected: str) -> str:
        return expected.split("text")[-1].strip().strip("'\"")

    def _extract_expected_url(self, expected: str) -> str:
        return expected.split()[-1]

    def _extract_expected_title(self, expected: str) -> str:
        return expected.split("title")[-1].strip().strip("'\"")

    def _extract_expected_value(self, expected: str) -> str:
        return expected.split("value")[-1].strip().strip("'\"")
