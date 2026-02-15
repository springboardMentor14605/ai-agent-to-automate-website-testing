from playwright.sync_api import sync_playwright, expect
import re
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from datetime import datetime

os.environ["GOOGLE_API_KEY"] = ""

class PlaywrightExecutor:
    """Executes generated Playwright actions and assertions in a headless browser environment."""

    def __init__(self, headless: bool = True, use_llm_debugging: bool = False):
        # Sets whether the browser will run with a visible UI or in the background
        self.headless = headless
        self.use_llm_debugging = use_llm_debugging
        
        # Initialize Gemini for debugging assistance if enabled
        if self.use_llm_debugging:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview",
                    temperature=0.3,
                    google_api_key=google_api_key
                )
            else:
                print("⚠️  Warning: GOOGLE_API_KEY not found. LLM debugging disabled.")
                self.use_llm_debugging = False

    def execute_test(self, test_steps: List[Dict[str, Any]], test_name: str = "Test Suite") -> Dict[str, Any]:
        """
        Execute a list of test steps and return detailed results.
        
        Args:
            test_steps: List of test step dictionaries
            test_name: Name of the test suite for reporting
            
        Returns:
            Dictionary containing test execution results
        """
        # Print header
        self._print_header(test_name, len(test_steps))
        
        # Dictionary to track the outcome of the test run
        results = {
            "test_name": test_name,
            "status": "PASS",
            "steps_executed": 0,
            "steps_passed": 0,
            "steps_failed": 0,
            "total_steps": len(test_steps),
            "error": None,
            "failed_step": None,
            "failed_step_details": None,
            "execution_time": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        start_time = datetime.now()

        with sync_playwright() as p:
            # Launching the Chromium browser
            print(f"🌐 Launching {'headless' if self.headless else 'headed'} Chromium browser...")
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()
            print("✅ Browser launched successfully\n")

            try:
                for idx, step in enumerate(test_steps, 1):
                    # Run each individual action and its assertions
                    print(f"📍 Step {idx}/{len(test_steps)}: {self._format_step_description(step)}")
                    
                    try:
                        self._execute_step(page, step)
                        results["steps_executed"] += 1
                        results["steps_passed"] += 1
                        print(f"   ✅ Step {idx} completed successfully\n")
                    except Exception as step_error:
                        results["steps_failed"] += 1
                        results["failed_step"] = idx
                        results["failed_step_details"] = step
                        raise step_error
                        
            except Exception as e:
                # Capture errors if a step fails
                print(f"\n❌ Error executing step {results.get('failed_step', '?')}: {e}\n")
                results["status"] = "FAIL"
                results["error"] = str(e)
                
                # Use LLM to suggest debugging steps if enabled
                if self.use_llm_debugging and results["failed_step_details"]:
                    print("🤖 Analyzing failure with AI...\n")
                    suggestion = self._get_debug_suggestion(results["failed_step_details"], str(e))
                    results["debug_suggestion"] = suggestion
                    self._print_debug_suggestion(suggestion)
            finally:
                # Ensure resources are cleaned up
                print("🔒 Closing browser...")
                browser.close()
                
        # Calculate execution time
        end_time = datetime.now()
        results["execution_time"] = str(end_time - start_time)
        
        # Print summary
        self._print_summary(results)

        return results

    def _execute_step(self, page, step: Dict[str, Any]):
        """Execute a single test step with its actions and assertions."""
        action = step.get("action")
        target = step.get("target")
        value = step.get("value")
        
        # Some formats might put params in a 'params' dict
        if "params" in step:
            params = step["params"]
            target = params.get("target", target) or params.get("url", target) or params.get("selector", target)
            value = params.get("value", value)

        assertions = step.get("assertions", [])

        # Map abstract instructions to actual Playwright browser commands
        if action == "open":
            url = target if target.startswith("http") else f"https://{target}"
            print(f"   🔗 Opening: {url}")
            page.goto(url)
        elif action == "click":
            print(f"   👆 Clicking: {target}")
            # Wait for element to be actionable
            page.wait_for_selector(target, state="visible", timeout=5000)
            page.click(target)
        elif action == "fill":
            print(f"   ⌨️  Filling '{target}' with: {value}")
            page.wait_for_selector(target, state="visible", timeout=5000)
            page.fill(target, value)
        elif action == "wait":
            print(f"   ⏳ Waiting: {value}ms")
            page.wait_for_timeout(int(value))
        elif action == "assert":
            # Handled by assertions loop below
            pass
        else:
            print(f"   ⚠️  Unknown action: {action}")

        # Dynamically run the assertion code generated by the LLM
        if assertions:
            print(f"   🔍 Running {len(assertions)} assertion(s)...")
            
        for idx, assertion in enumerate(assertions, 1):
            # Simple security check to prevent completely arbitrary code
            if assertion.strip().startswith("expect("):
                try:
                    # using eval for single expression assertions
                    eval(assertion, {"expect": expect, "page": page, "re": re})
                    print(f"      ✓ Assertion {idx} passed: {self._truncate(assertion, 60)}")
                except AssertionError as e:
                    print(f"      ✗ Assertion {idx} failed: {self._truncate(assertion, 60)}")
                    raise AssertionError(f"Assertion failed: {assertion}") from e
                except Exception as e:
                    print(f"      ⚠️  Assertion {idx} error: {self._truncate(assertion, 60)}")
                    raise RuntimeError(f"Failed to evaluate assertion '{assertion}': {e}") from e

    def _get_debug_suggestion(self, failed_step: Dict[str, Any], error_message: str) -> str:
        """Use Gemini to provide debugging suggestions based on the failed step and error."""
        system_prompt = """You are an expert Playwright test automation debugger.
Analyze the failed test step and error, then provide concise, actionable debugging suggestions.
Focus on common issues like:
- Incorrect selectors (CSS, XPath, text selectors)
- Timing issues (elements not loaded, animations)
- Element visibility and state
- Incorrect assertions or expected values
- Authentication or navigation issues

Format your response as a numbered list of 2-4 specific, actionable suggestions.
Keep it concise and practical."""

        user_prompt = f"""Failed Test Step:
Action: {failed_step.get('action')}
Target: {failed_step.get('target')}
Value: {failed_step.get('value')}
Assertions: {failed_step.get('assertions', [])}

Error Message:
{error_message}

Provide specific debugging suggestions:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Extract content from response
            content = response.content
            if isinstance(content, list):
                if len(content) > 0 and isinstance(content[0], dict) and 'text' in content[0]:
                    return content[0]['text']
            return str(content)
        except Exception as e:
            return f"Could not generate debug suggestion: {e}"

    def _format_step_description(self, step: Dict[str, Any]) -> str:
        """Format a step into a readable description."""
        action = step.get("action", "unknown")
        target = step.get("target", "")
        value = step.get("value", "")
        
        if "params" in step:
            params = step["params"]
            target = params.get("target", target) or params.get("url", target) or params.get("selector", target)
            value = params.get("value", value)
        
        if action == "open":
            return f"Open '{target}'"
        elif action == "click":
            return f"Click '{target}'"
        elif action == "fill":
            return f"Fill '{target}' with '{value}'"
        elif action == "wait":
            return f"Wait {value}ms"
        else:
            return f"{action.capitalize()} on '{target}'"

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max_length with ellipsis."""
        return text if len(text) <= max_length else text[:max_length-3] + "..."

    def _print_header(self, test_name: str, total_steps: int):
        """Print a formatted test header."""
        print("\n" + "="*70)
        print(f"🧪 TEST EXECUTION: {test_name}")
        print(f"📊 Total Steps: {total_steps}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

    def _print_debug_suggestion(self, suggestion: str):
        """Print formatted debug suggestions."""
        print("┌" + "─"*68 + "┐")
        print("│ 🤖 AI DEBUG SUGGESTIONS" + " "*44 + "│")
        print("├" + "─"*68 + "┤")
        for line in suggestion.split('\n'):
            if line.strip():
                # Wrap long lines
                if len(line) > 66:
                    words = line.split()
                    current_line = "│ "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 67:
                            current_line += word + " "
                        else:
                            print(current_line.ljust(68) + "│")
                            current_line = "│ " + word + " "
                    if current_line.strip() != "│":
                        print(current_line.ljust(68) + "│")
                else:
                    print(f"│ {line.ljust(66)} │")
        print("└" + "─"*68 + "┘\n")

    def _print_summary(self, results: Dict[str, Any]):
        """Print a formatted test summary."""
        print("\n" + "="*70)
        print("📋 TEST EXECUTION SUMMARY")
        print("="*70)
        
        status_icon = "✅" if results["status"] == "PASS" else "❌"
        print(f"{status_icon} Status: {results['status']}")
        print(f"📊 Steps Executed: {results['steps_executed']}/{results['total_steps']}")
        print(f"✅ Passed: {results['steps_passed']}")
        print(f"❌ Failed: {results['steps_failed']}")
        print(f"⏱️  Execution Time: {results['execution_time']}")
        
        if results["error"]:
            print(f"\n❌ Error Details:")
            print(f"   Step {results['failed_step']}: {results['error']}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    try:
        # Enable LLM debugging for intelligent error suggestions
        executor = PlaywrightExecutor(headless=False, use_llm_debugging=True)
        
        # Sample steps similar to what test_case_parser.py + assertion_generator.py would produce
        sample_steps = [
            {
                "action": "open",
                "target": "https://practice.automationtesting.in/",
                "assertions": [
                    "expect(page).to_have_title(re.compile('Automation Practice Site'))"
                ]
            },
            {
                "action": "click",
                "target": "#menu-item-40",  # Shop link
                "assertions": [
                    "expect(page).to_have_url(re.compile('.*shop.*'))"
                ]
            },
            {
                "action": "fill",
                "params": {
                    "selector": "input[name='s']",
                    "value": "selenium"
                },
                "assertions": []
            }
        ]
        
        # Execute the test
        result = executor.execute_test(sample_steps, test_name="Sample E-Commerce Test")
        
        # You can also access results programmatically
        if result["status"] == "FAIL":
            print(f"⚠️  Test failed at step {result['failed_step']}")
            
    except Exception as e:
        print(f"💥 Critical error: {e}")
