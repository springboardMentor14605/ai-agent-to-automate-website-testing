import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class PlaywrightAssertionAI:
    """Generates Playwright assertions from plain English using an LLM."""

    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("Missing Hugging Face API token in .env file")

        model = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.2,
            max_new_tokens=500,
            huggingfacehub_api_token=token
        )

        self.chat = ChatHuggingFace(llm=model)

    def create_assertions(self, step_data: Dict) -> List[str]:
        expected_text = step_data.get("expected")

        if not expected_text:
            return []

        system_msg = SystemMessage(
            content="You are a Playwright Python expert. "
                    "Generate only valid expect() assertions. "
                    "No explanations. No markdown."
        )

        user_msg = HumanMessage(
            content=f"""
Test expectation:
{expected_text}

Rules:
- Use Playwright Python syntax
- One assertion per line
- Only output code starting with expect(
"""
        )

        try:
            reply = self.chat.invoke([system_msg, user_msg])
            return self._clean_output(reply.content)
        except Exception as error:
            print("Generation failed:", error)
            return []

    def _clean_output(self, text: str) -> List[str]:
        text = text.replace("```python", "").replace("```", "").strip()
        lines = text.split("\n")

        valid_assertions = []
        for line in lines:
            line = line.strip()
            if line.startswith("expect("):
                valid_assertions.append(line)

        return valid_assertions


if __name__ == "__main__":
    generator = PlaywrightAssertionAI()

    example_step = {
        "action": "click",
        "target": "submit-button",
        "expected": "The message 'Form submitted successfully' should be visible"
    }

    results = generator.create_assertions(example_step)

    print("Generated Assertions:")
    for r in results:
        print(r)
