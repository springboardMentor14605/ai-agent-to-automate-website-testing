from typing import Dict, List, Optional
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
os.environ["GOOGLE_API_KEY"] = ""

class LLMAssertionGenerator:
    """Uses LLM to generate Playwright assertions from natural language expectations."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        # Configure the Gemini model
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in .env file.")

        self.chat_model = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,  # Very low temperature for consistent code generation
            google_api_key=google_api_key
        )

    def generate_assertions(self, parsed_step: Dict) -> List[str]:
        """
        Generate Playwright assertions from parsed step.
        Returns a list of clean, executable assertion strings.
        """
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
            # Invoke the Gemini model
            response = self.chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Extract content from response
            content = self._extract_content(response.content)
            
            # Clean and validate the output
            assertions = self._post_process(content)
            
            # Validate assertions before returning
            return self._validate_assertions(assertions)
            
        except Exception as e:
            print(f"Error generating assertions: {e}")
            return []

    def _extract_content(self, content) -> str:
        """
        Extract text content from various Gemini response formats.
        """
        # If content is a list with text field
        if isinstance(content, list):
            if len(content) > 0 and isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
            # If it's just a list of strings, join them
            elif all(isinstance(item, str) for item in content):
                return '\n'.join(content)
            else:
                return str(content)
        
        # If content is already a string
        if isinstance(content, str):
            return content
        
        # Fallback
        return str(content)

    def _build_prompts(self, action: str, target: str, expected: str):
        """
        Build system and user prompts for assertion generation.
        """
        system_prompt = """You are an expert QA automation engineer specializing in Playwright (Python).

Your task: Generate ONLY valid Playwright assertion statements.

CRITICAL RULES:
1. Output ONLY executable Python code - NO markdown, NO explanations, NO comments
2. Each assertion must start with "expect("
3. One assertion per line
4. Use proper Playwright assertion methods
5. Do NOT include imports, function definitions, or any other code

Output format example:
expect(page.locator("text=Submit")).to_be_visible()
expect(page).to_have_title(re.compile("Dashboard"))
"""
        
        user_prompt = f"""Generate Playwright assertions for:

Action: {action}
Target: {target}
Expected: {expected}

Available assertion methods:
- to_be_visible()
- to_be_hidden()
- to_be_enabled()
- to_be_disabled()
- to_contain_text("text")
- to_have_text("text")
- to_have_value("value")
- to_have_title(re.compile("pattern"))
- to_have_url(re.compile("pattern"))
- to_be_checked()
- to_have_count(number)

Generate ONLY the assertion code, nothing else:"""
        
        return system_prompt, user_prompt

    def _post_process(self, llm_output: str) -> List[str]:
        """
        Clean and extract valid assertions from LLM output.
        """
        # Remove common markdown patterns
        cleaned = llm_output
        
        # Remove code block markers
        cleaned = re.sub(r'```python\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Remove common prefixes
        cleaned = re.sub(r'^(Output:|Generated assertions?:|Code:)\s*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Split into lines
        lines = cleaned.split('\n')
        
        assertions = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Only keep lines that start with 'expect('
            if line.startswith('expect('):
                # Clean up any trailing artifacts
                line = line.rstrip(';').rstrip(',').strip()
                assertions.append(line)
        
        return assertions

    def _validate_assertions(self, assertions: List[str]) -> List[str]:
        """
        Validate that assertions are properly formatted.
        Returns only valid assertions.
        """
        valid_assertions = []
        
        # Common Playwright assertion methods
        valid_methods = [
            'to_be_visible', 'to_be_hidden', 'to_be_enabled', 'to_be_disabled',
            'to_contain_text', 'to_have_text', 'to_have_value', 'to_have_title',
            'to_have_url', 'to_be_checked', 'to_have_count', 'to_have_attribute',
            'to_have_class', 'to_have_css', 'to_have_id', 'to_be_attached',
            'to_be_editable', 'to_be_empty', 'to_be_focused', 'to_be_in_viewport'
        ]
        
        for assertion in assertions:
            # Check if it's a valid expect statement
            if not assertion.startswith('expect('):
                continue
            
            # Check if it contains a valid method
            has_valid_method = any(method in assertion for method in valid_methods)
            
            if has_valid_method:
                # Ensure balanced parentheses
                if assertion.count('(') == assertion.count(')'):
                    valid_assertions.append(assertion)
                else:
                    print(f"Warning: Skipping malformed assertion: {assertion}")
        
        return valid_assertions

    def format_output(self, assertions: List[str]) -> str:
        """
        Format assertions as readable code block.
        """
        if not assertions:
            return "# No assertions generated"
        
        return '\n'.join(assertions)


if __name__ == "__main__":
    try:
        generator = LLMAssertionGenerator()
        
        # Test cases
        test_cases = [
            {
                "action": "click",
                "target": "submit-button",
                "expected": "The success message 'Form submitted' should be visible"
            },
            {
                "action": "fill",
                "target": "username-input",
                "expected": "The username field should contain 'testuser'"
            },
            {
                "action": "navigate",
                "target": "dashboard",
                "expected": "Page title should contain 'Dashboard' and URL should include '/dashboard'"
            }
        ]
        
        for i, sample_step in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}:")
            print(f"Action: {sample_step['action']}")
            print(f"Target: {sample_step['target']}")
            print(f"Expected: {sample_step['expected']}")
            print(f"{'='*60}")
            
            assertions = generator.generate_assertions(sample_step)
            
            if assertions:
                print("\nGenerated Assertions:")
                print(generator.format_output(assertions))
            else:
                print("\nNo valid assertions generated")
            
    except Exception as e:
        print(f"Setup failed: {e}")