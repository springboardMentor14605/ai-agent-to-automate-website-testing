import os
from typing import Dict, List


def extract_text(content) -> str:
    """Safely extract plain text from Gemini response content (string or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


class LLMAssertionGenerator:
    """Uses Gemini to generate Playwright assertions from natural language expectations."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API key is required.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=api_key,
            temperature=0
        )

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        expected = parsed_step.get("expected")
        target   = parsed_step.get("target")
        action   = parsed_step.get("action", "N/A")

        if not expected:
            return []

        from langchain_core.messages import HumanMessage
        response   = self.llm.invoke([HumanMessage(content=self._build_prompt(action, target, expected))])
        assertions = self._post_process(extract_text(response.content))

        if action == "fill" and assertions:
            assertion = assertions[0]
            if "to_have_value(" in assertion:
                value = assertion.split("to_have_value(")[1].split(")")[0].strip("\"'")
                parsed_step["value"] = value

        return assertions

    def _build_prompt(self, action: str, target: str, expected: str) -> str:
        return f"""You are an expert QA automation engineer.
Generate ONLY a single valid Playwright (Python) assertion. No explanation, no imports, no markdown.
- Action: {action}
- Target: {target}
- Expected: {expected}
Rules:
- Use expect() Python syntax only
- For errors always use selector "#error"
- Output one assertion line only
Now generate:"""

    def _post_process(self, text: str) -> List[str]:
        return [line.strip() for line in text.split("\n") if line.strip().startswith("expect(")]