from typing import Dict, List, Optional
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMAssertionGenerator:
    """Uses LLM to generate Playwright assertions from natural language expectations."""

    def __init__(self, repo_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        # Configure the model
        # Use HuggingFace endpoint as per other assignments
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HuggingFace API Token not found. Please set HUGGINGFACEHUB_API_TOKEN in .env file.")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=1000,
            temperature=0.2, # Low temperature for consistent code generation
            huggingfacehub_api_token=hf_token
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        # Handle cases where keys might be directly in parsed_step or nested in 'params'
        expected = parsed_step.get("expected")
        target = parsed_step.get("target")
        action = parsed_step.get("action")
        
        # Fallback to checking inside 'params' if available
        if not expected and "params" in parsed_step:
            params = parsed_step["params"]
            expected = params.get("expected")
            target = params.get("target") or target

        if not expected:
            return []

        # Convert parsed step into a structured prompt
        system_prompt, user_prompt = self._build_prompts(action, target, expected)
        
        try:
            # Invoke the Llama model
            response = self.chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            # Clean and validate the output
            return self._post_process(response.content)
        except Exception as e:
            print(f"Error generating assertions: {e}")
            return []

    def _build_prompts(self, action: str, target: str, expected: str):
        # Specific system prompt for code generation
        system_prompt = """You are an expert QA automation engineer specializing in Playwright (Python).
Your task is to generate ONLY valid Playwright assertion statements based on the provided context.
Output purely the code. Do not include markdown formatting, explanations, or imports.
"""
        
        user_prompt = f"""
Context:
Action performed: {action}
Target element: {target}
Expected outcome: {expected}

Rules:
- Generate Playwright Python assertions using `expect()`.
- One assertion per line.
- Do NOT add comments or explanations.
- Do NOT wrap in markdown blocks.

Examples:
Input: Expected 'Submit' button to be visible
Output: expect(page.locator("text=Submit")).to_be_visible()

Input: Expected page title to contain 'Dashboard'
Output: expect(page).to_have_title(re.compile("Dashboard"))

Now generate assertions for the expected outcome: {expected}
"""
        return system_prompt, user_prompt

    def _post_process(self, llm_output: str) -> List[str]:
        # Remove markdown code blocks and clean up
        cleaned_output = llm_output.replace("```python", "").replace("```", "").strip()
        
        lines = cleaned_output.split("\n")
        assertions = []

        for line in lines:
            line = line.strip()
            # Ensure only valid Playwright code is kept
            if line.startswith("expect("):
                assertions.append(line)

        return assertions

if __name__ == "__main__":
    try:
        generator = LLMAssertionGenerator()
        
        sample_step = {
            "action": "click",
            "target": "submit-button",
            "expected": "The success message 'Form submitted' should be visible"
        }
        
        print(f"Generating assertions for: {sample_step}")
        assertions = generator.generate_assertions(sample_step)
        
        print("\nGenerated Assertions:")
        for assertion in assertions:
            print(assertion)
            
    except Exception as e:
        print(f"Setup failed: {e}")
