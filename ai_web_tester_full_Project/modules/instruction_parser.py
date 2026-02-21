"""
Instruction Parser Module
Converts natural language test descriptions into structured browser action commands.
"""

import os
import json
import re
from typing import List, Dict, Union, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are an instruction parser for an automated web testing system.

Your task: Convert a natural language test case into structured test commands.

Output ONLY valid JSON — no explanations, no markdown, no extra keys.

Allowed actions:
- open      : navigate to a URL
- fill      : type text into an input field
- click     : click a button, link, or element
- select    : choose an option from a dropdown
- hover     : hover over an element
- scroll    : scroll to an element
- wait      : wait for a specified time (ms)
- clear     : clear an input field
- assert    : make an assertion about the page state

Output format:
[
  {
    "action": "<action>",
    "params": {
      "target": "<CSS selector or URL>",
      "value": "<value if applicable>",
      "expected": "<expected state for assertions>"
    },
    "description": "<human readable step description>"
  }
]

Guidelines for selectors:
- Use CSS selectors: #id, .class, [attribute=value], tag
- For URLs use the full URL (e.g. https://example.com)
- For text-based clicks use: text=Submit or button:has-text("Submit")
- Prefer IDs when possible (#username, #password, #login-button)
"""


class InstructionParser:
    """Parses natural language test instructions into structured action commands."""

    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    def parse(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Parse natural language instruction into a list of structured action steps.

        Args:
            instruction: Natural language test case description.

        Returns:
            List of action dictionaries.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Parse the following test case into structured actions:\n\n{instruction}")
        ]

        try:
            response = self.llm.invoke(messages)
            raw = self._extract_text(response.content)
            cleaned = self._clean_json(raw)
            steps = json.loads(cleaned)

            if not isinstance(steps, list):
                raise ValueError("Expected a JSON array of steps.")

            return self._normalize_steps(steps)

        except json.JSONDecodeError as e:
            print(f"[InstructionParser] JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[InstructionParser] Error: {e}")
            return []

    def _extract_text(self, content) -> str:
        """Normalize LLM response content to a plain string."""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and whitespace from LLM output."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()

    def _normalize_steps(self, steps: List[Dict]) -> List[Dict]:
        """Ensure every step has required keys with safe defaults."""
        normalized = []
        for i, step in enumerate(steps):
            params = step.get("params", {})
            normalized.append({
                "step_number": i + 1,
                "action": step.get("action", "").lower().strip(),
                "target": params.get("target") or step.get("target", ""),
                "value": params.get("value") or step.get("value", ""),
                "expected": params.get("expected") or step.get("expected", ""),
                "description": step.get("description", f"Step {i + 1}: {step.get('action', '')}"),
                "params": params,
                "assertions": step.get("assertions", [])
            })
        return normalized

    def format_steps(self, steps: List[Dict]) -> str:
        """Return a human-readable summary of parsed steps."""
        if not steps:
            return "No steps parsed."
        lines = ["Parsed Test Steps:", "=" * 40]
        for step in steps:
            lines.append(f"  {step['step_number']}. [{step['action'].upper()}] {step['description']}")
            if step["target"]:
                lines.append(f"     Target : {step['target']}")
            if step["value"]:
                lines.append(f"     Value  : {step['value']}")
            if step["expected"]:
                lines.append(f"     Expect : {step['expected']}")
        return "\n".join(lines)
