from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class LLMAssertionGenerator:
    """
    Uses Gemini LLM to generate Playwright assertions
    from natural language expectations.
    """
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        expected = parsed_step.get("expected")
        target = parsed_step.get("target")
        action = parsed_step.get("action")

        if not expected:
            return []

        prompt = self._build_prompt(action, target, expected)
        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])
        return self._post_process(response.content)

    def _build_prompt(self, action: str, target: str, expected: str) -> str:
        return f"""
You are an expert QA automation engineer.
Your task is to generate ONLY valid Playwright (Python) assertions.

Context:
- Action performed: {action}
- Target element: {target}
- Expected outcome: {expected}

Rules:
- Output ONLY assertion statements
- Do NOT include explanations
- Do NOT include imports
- Do NOT include markdown
- Use Playwright expect() syntax
- One assertion per line

Now generate the assertions:
"""

    def _post_process(self, llm_output: str) -> List[str]:
        lines = llm_output.split("\n")
        assertions = []
        for line in lines:
            line = line.strip()
            if line.startswith("expect("):
                assertions.append(line)
        return assertions
