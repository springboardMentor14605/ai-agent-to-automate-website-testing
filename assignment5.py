from typing import Dict, List
import re

class AssertionGenerator:
    """
    Generates Playwright assertions based on 
    parsed expected outcomes from natural language.
    """

    def __init__(self):
        # Keywords used to identify the type of validation needed
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

        # Rule-based mapping: checks keywords to decide which Playwright assertion to write
        
        # 1. Visibility
        if any(word in expected_lower for word in self.visibility_keywords):
            assertions.append(f"expect(page.locator('{target}')).to_be_visible()")

        # 2. Hidden
        elif any(word in expected_lower for word in self.hidden_keywords):
            assertions.append(f"expect(page.locator('{target}')).to_be_hidden()")

        # 3. Text (Exact match or Contains)
        elif any(word in expected_lower for word in self.text_keywords):
            expected_text = self._extract_quoted_value(expected)
            if expected_text:
                if "contain" in expected_lower:
                    assertions.append(f"expect(page.locator('{target}')).to_contain_text('{expected_text}')")
                else:
                    assertions.append(f"expect(page.locator('{target}')).to_have_text('{expected_text}')")

        # 4. URL
        elif any(word in expected_lower for word in self.url_keywords):
            expected_url = self._extract_url_value(expected)
            if expected_url:
                # Use regex for flexible URL matching if not a full URL
                if expected_url.startswith("http"):
                    assertions.append(f"expect(page).to_have_url('{expected_url}')")
                else:
                    assertions.append(f"expect(page).to_have_url(re.compile('{expected_url}'))")

        # 5. Title
        elif any(word in expected_lower for word in self.title_keywords):
            expected_title = self._extract_quoted_value(expected)
            if expected_title:
                assertions.append(f"expect(page).to_have_title('{expected_title}')")

        # 6. Input Value
        elif any(word in expected_lower for word in self.value_keywords):
            expected_value = self._extract_quoted_value(expected)
            if expected_value:
                assertions.append(f"expect(page.locator('{target}')).to_have_value('{expected_value}')")
        
        # 7. Enabled/Disabled
        elif "enabled" in expected_lower or "clickable" in expected_lower:
            assertions.append(f"expect(page.locator('{target}')).to_be_enabled()")
        elif "disabled" in expected_lower:
            assertions.append(f"expect(page.locator('{target}')).to_be_disabled()")

        return assertions

    # Helper methods to parse the specific expected string
    def _extract_quoted_value(self, text: str) -> str:
        """Extracts text inside single or double quotes."""
        match = re.search(r"['\"](.*?)['\"]", text)
        return match.group(1) if match else ""

    def _extract_url_value(self, text: str) -> str:
        """Extracts a URL or path from the text."""
        # First try to find a quoted URL
        quoted = self._extract_quoted_value(text)
        if quoted:
            return quoted
        
        # Fallback: simple split if it looks like a url/path (contains / or http)
        words = text.split()
        for word in words:
            if "/" in word or "http" in word:
                return word.strip("'\",")
        return ""

if __name__ == "__main__":
    generator = AssertionGenerator()
    
    test_cases = [
        {"target": "#submit", "expected": "The button should be visible"},
        {"target": "#error", "expected": "The error message should be hidden"},
        {"target": "#title", "expected": "The page title should be 'My Page'"},
        {"target": "#link", "expected": "The url should contain '/dashboard'"},
        {"target": "#input", "expected": "The input value should be 'john_doe'"},
        {"target": "#btn", "expected": "The button should be enabled"}
    ]
    
    for case in test_cases:
        assertions = generator.generate_assertions(case)
        print(f"Goal: {case['expected']} -> Assertions: {assertions}")
