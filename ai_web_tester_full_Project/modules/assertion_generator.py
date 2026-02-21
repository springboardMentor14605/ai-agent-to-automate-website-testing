"""
Code Generation Module
Converts parsed action steps into executable Playwright Python assertion strings.
"""

import os
import re
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


VALID_ASSERTION_METHODS = [
    "to_be_visible", "to_be_hidden", "to_be_enabled", "to_be_disabled",
    "to_contain_text", "to_have_text", "to_have_value", "to_have_title",
    "to_have_url", "to_be_checked", "to_have_count", "to_have_attribute",
    "to_have_class", "to_have_css", "to_have_id", "to_be_attached",
    "to_be_editable", "to_be_empty", "to_be_focused", "to_be_in_viewport",
    "not_to_be_visible", "not_to_contain_text", "not_to_have_text"
]

SYSTEM_PROMPT = """You are an expert QA automation engineer specializing in Playwright (Python).

Your task: Generate ONLY valid Playwright Python assertion statements.

CRITICAL RULES:
1. Output ONLY executable Python code — NO markdown, NO explanations, NO comments
2. Every assertion MUST start exactly with: expect(
3. One assertion per line
4. Use proper Playwright assertion methods
5. Do NOT include imports, function definitions, or any other code
6. Use re.compile() for regex URL/title patterns

Available assertion methods:
  expect(page).to_have_title(re.compile("pattern"))
  expect(page).to_have_url(re.compile("pattern"))
  expect(page.locator("selector")).to_be_visible()
  expect(page.locator("selector")).to_be_hidden()
  expect(page.locator("selector")).to_be_enabled()
  expect(page.locator("selector")).to_be_disabled()
  expect(page.locator("selector")).to_contain_text("text")
  expect(page.locator("selector")).to_have_text("text")
  expect(page.locator("selector")).to_have_value("value")
  expect(page.locator("selector")).to_have_count(n)
  expect(page.locator("selector")).to_be_checked()
  expect(page.locator("selector")).to_have_attribute("attr", "value")

Output format example:
expect(page.locator("text=Submit")).to_be_visible()
expect(page).to_have_title(re.compile("Dashboard"))
"""


class AssertionGenerator:
    """Generates Playwright assertions from natural language expected outcomes."""

    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.1):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    def generate(self, step: Dict) -> List[str]:
        """
        Generate Playwright assertions for a parsed test step.

        Args:
            step: Parsed step dictionary with action, target, expected fields.

        Returns:
            List of valid Playwright assertion strings.
        """
        expected = step.get("expected") or ""
        if not expected:
            # Also check nested params
            params = step.get("params", {})
            expected = params.get("expected", "")

        if not expected:
            return []

        action = step.get("action", "unknown")
        target = step.get("target", "")

        user_prompt = f"""Generate Playwright (Python) assertions for this test step:

Action   : {action}
Target   : {target}
Expected : {expected}

Generate ONLY assertion code, nothing else:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            raw = self._extract_text(response.content)
            assertions = self._post_process(raw)
            return self._validate(assertions)
        except Exception as e:
            print(f"[AssertionGenerator] Error: {e}")
            return []

    def generate_for_steps(self, steps: List[Dict]) -> List[Dict]:
        """Enrich each step with generated assertions."""
        enriched = []
        for step in steps:
            if not step.get("assertions"):
                step["assertions"] = self.generate(step)
            enriched.append(step)
        return enriched

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_text(self, content) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _post_process(self, llm_output: str) -> List[str]:
        """Strip markdown and extract lines that start with expect(."""
        cleaned = re.sub(r"```(?:python)?\s*", "", llm_output)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = re.sub(
            r"^(Output:|Generated assertions?:|Code:|Assertions?:)\s*",
            "", cleaned, flags=re.IGNORECASE | re.MULTILINE
        )

        assertions = []
        for line in cleaned.split("\n"):
            line = line.strip().rstrip(";").rstrip(",")
            if line.startswith("expect(") and line:
                assertions.append(line)
        return assertions

    def _validate(self, assertions: List[str]) -> List[str]:
        """Keep only assertions with recognised methods and balanced parens."""
        valid = []
        for a in assertions:
            if not a.startswith("expect("):
                continue
            has_method = any(m in a for m in VALID_ASSERTION_METHODS)
            balanced = a.count("(") == a.count(")")
            if has_method and balanced:
                valid.append(a)
            else:
                print(f"[AssertionGenerator] Skipping malformed assertion: {a}")
        return valid
