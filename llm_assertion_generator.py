from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json
import os

class LLMAssertionGenerator:
    """
    Uses Gemini LLM to generate Playwright assertions
    from natural language expectations.
    """

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=api_key,
            temperature=0
        )

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        expected = parsed_step.get("expected")
        target = parsed_step.get("target")
        action = parsed_step.get("action", "N/A")

        if not expected:
            return []

        prompt = self._build_prompt(action, target, expected)

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])
        assertions = self._post_process(response.content)

        # AUTO-EXTRACT VALUE FOR FILL ACTIONS
        if action == "fill" and assertions:
            assertion = assertions[0]
            if "to_have_value(" in assertion:
                value = (
                    assertion
                    .split("to_have_value(")[1]
                    .split(")")[0]
                    .strip("\"'")
                )
                parsed_step["value"] = value

        return assertions

    def _build_prompt(self, action: str, target: str, expected: str) -> str:
        return f"""
You are an expert QA automation engineer.

Your task is to generate ONLY valid Playwright (Python) assertions.
DO NOT generate JavaScript syntax.

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
- If multiple assertions are possible, output only the single most appropriate assertion
- If expected mentions error, error message, or login error:
  ALWAYS use selector "#error"
  NEVER use any other selector for errors
- One assertion per line

Now generate the assertions:
"""

    def _post_process(self, llm_output) -> List[str]:
        assertions = []

        if isinstance(llm_output, list):
            text_output = ""
            for item in llm_output:
                if isinstance(item, dict) and "text" in item:
                    text_output += item["text"] + "\n"
        else:
            text_output = llm_output

        for line in text_output.split("\n"):
            line = line.strip()
            if line.startswith("expect("):
                assertions.append(line)

        return assertions

# EXECUTION
if __name__ == "__main__":    
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set as an environment variable")
    
    generator = LLMAssertionGenerator(API_KEY)

    test_cases = [

    # Open login page
    {
        "action": "open",
        "target": "https://practicetestautomation.com/practice-test-login/",
        "expected": "Username, password and submit button should be visible"
    },

    # Submit without credentials
    {
        "action": "click",
        "target": "#submit",
        "expected": "Error message should be visible"
    },

    # Enter username
    {
        "action": "fill",
        "target": "#username",
        "expected": "Username field should contain student"
    },

    # Enter password
    {
        "action": "fill",
        "target": "#password",
        "expected": "Password field should contain Password123"
    },

    # Submit login
    {
        "action": "click",
        "target": "#submit",
        "expected": "User should be redirected to https://practicetestautomation.com/logged-in-successfully/"
    },

    # Success page validations
    {
        "action": "wait",
        "target": "page",
        "expected": "Logged In Successfully text should be visible"
    },
    {
        "action": "wait",
        "target": "page",
        "expected": "Congratulations student. You successfully logged in! should be visible"
    }
]

    enriched_steps = []

    for index, step in enumerate(test_cases, start=1):
        print(f"\nTest Case {index}")
        print("Input:", step)

        assertions = generator.generate_assertions(step)

        step_with_assertions = {
            "action": step.get("action"),
            "target": step.get("target"),
            "value": step.get("value"),
            "expected": step.get("expected"),
            "assertions": assertions
        }

        enriched_steps.append(step_with_assertions)

        for assertion in assertions:
            print("Generated Assertion:", assertion)

    # SAVE OUTPUT
    with open("llm_generated_steps.json", "w") as f:
        json.dump(enriched_steps, f, indent=2)

    print("\n Assertions saved to llm_generated_steps.json")