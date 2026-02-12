from typing import Dict, List
import re


class SimpleAssertionBuilder:
    """Creates Playwright assertions from plain English expectations."""

    def build(self, step: Dict) -> List[str]:
        result = []

        description = step.get("expected", "")
        element = step.get("target", "")

        if not description:
            return result

        text = description.lower()

        # Visible
        if "visible" in text or "shown" in text:
            result.append(f"expect(page.locator('{element}')).to_be_visible()")

        # Hidden
        elif "hidden" in text or "not visible" in text:
            result.append(f"expect(page.locator('{element}')).to_be_hidden()")

        # Contains text
        elif "contain" in text:
            value = self._get_text_inside_quotes(description)
            if value:
                result.append(
                    f"expect(page.locator('{element}')).to_contain_text('{value}')"
                )

        # Exact text
        elif "text" in text or "message" in text:
            value = self._get_text_inside_quotes(description)
            if value:
                result.append(
                    f"expect(page.locator('{element}')).to_have_text('{value}')"
                )

        # URL check
        elif "url" in text:
            value = self._get_text_inside_quotes(description)
            if value:
                if value.startswith("http"):
                    result.append(f"expect(page).to_have_url('{value}')")
                else:
                    result.append(f"expect(page).to_have_url(re.compile('{value}'))")

        # Page title
        elif "title" in text:
            value = self._get_text_inside_quotes(description)
            if value:
                result.append(f"expect(page).to_have_title('{value}')")

        # Input value
        elif "value" in text or "input" in text:
            value = self._get_text_inside_quotes(description)
            if value:
                result.append(
                    f"expect(page.locator('{element}')).to_have_value('{value}')"
                )

        # Enabled / Disabled
        elif "enabled" in text or "clickable" in text:
            result.append(f"expect(page.locator('{element}')).to_be_enabled()")

        elif "disabled" in text:
            result.append(f"expect(page.locator('{element}')).to_be_disabled()")

        return result

    def _get_text_inside_quotes(self, sentence: str) -> str:
        match = re.search(r"['\"](.*?)['\"]", sentence)
        return match.group(1) if match else ""


if __name__ == "__main__":
    builder = SimpleAssertionBuilder()

    examples = [
        {"target": "#submit", "expected": "The button should be visible"},
        {"target": "#error", "expected": "The error message should be hidden"},
        {"target": "#title", "expected": "The page title should be 'My Page'"},
        {"target": "#link", "expected": "The url should contain '/dashboard'"},
        {"target": "#input", "expected": "The input value should be 'john_doe'"},
        {"target": "#btn", "expected": "The button should be enabled"},
    ]

    for example in examples:
        print(builder.build(example))
