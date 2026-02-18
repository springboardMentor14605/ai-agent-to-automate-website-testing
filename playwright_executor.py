from playwright.sync_api import sync_playwright, expect
import re
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
os.environ["GOOGLE_API_KEY"] = ""  # Add your key here

class PlaywrightExecutor:
    """Executes generated Playwright actions and assertions in a headless browser environment."""

    def __init__(self, headless: bool = True, use_llm_debugging: bool = False):
        self.headless = headless
        self.use_llm_debugging = use_llm_debugging
        
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
        self._print_header(test_name, len(test_steps))
        
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
            print(f"🌐 Launching {'headless' if self.headless else 'headed'} Chromium browser...")
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()
            print("✅ Browser launched successfully\n")

            try:
                for idx, step in enumerate(test_steps, 1):
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
                print(f"\n❌ Error executing step {results.get('failed_step', '?')}: {e}\n")
                results["status"] = "FAIL"
                results["error"] = str(e)
                if self.use_llm_debugging and results["failed_step_details"]:
                    print("🤖 Analyzing failure with AI...\n")
                    suggestion = self._get_debug_suggestion(results["failed_step_details"], str(e))
                    results["debug_suggestion"] = suggestion
                    self._print_debug_suggestion(suggestion)
            finally:
                print("🔒 Closing browser...")
                browser.close()

        end_time = datetime.now()
        results["execution_time"] = str(end_time - start_time)
        self._print_summary(results)
        return results

    def _execute_step(self, page, step: Dict[str, Any]):
        action = step.get("action")
        target = step.get("target")
        value = step.get("value")

        if "params" in step:
            params = step["params"]
            target = params.get("target", target) or params.get("url", target) or params.get("selector", target)
            value = params.get("value", value)

        assertions = step.get("assertions", [])

        if action == "open":
            url = target if target.startswith("http") else f"https://{target}"
            print(f"   🔗 Opening: {url}")
            page.goto(url)
            page.wait_for_load_state("networkidle")
        elif action == "fill":
            print(f"   ⌨️  Filling '{target}' with: '{value}'")
            page.wait_for_selector(target, state="visible", timeout=10000)
            page.fill(target, value)
        elif action == "click":
            print(f"   👆 Clicking: {target}")
            page.wait_for_selector(target, state="visible", timeout=10000)
            page.wait_for_load_state("domcontentloaded")
            page.click(target)
        elif action == "select":
            print(f"   📋 Selecting '{value}' from: {target}")
            page.wait_for_selector(target, state="visible", timeout=10000)
            page.select_option(target, value)
            page.wait_for_load_state("domcontentloaded")
        elif action == "wait":
            print(f"   ⏳ Waiting: {value}ms")
            page.wait_for_timeout(int(value))
        elif action == "scroll":
            print(f"   📜 Scrolling to: {target}")
            page.locator(target).scroll_into_view_if_needed()
        elif action == "hover":
            print(f"   🖱️  Hovering over: {target}")
            page.wait_for_selector(target, state="visible", timeout=10000)
            page.hover(target)
        elif action == "clear":
            print(f"   🧹 Clearing: {target}")
            page.wait_for_selector(target, state="visible", timeout=10000)
            page.fill(target, "")

        if assertions:
            print(f"   🔍 Running {len(assertions)} assertion(s)...")

        for idx, assertion in enumerate(assertions, 1):
            if assertion.strip().startswith("expect("):
                try:
                    eval(assertion, {"expect": expect, "page": page, "re": re})
                    print(f"      ✓ Assertion {idx} passed: {self._truncate(assertion, 65)}")
                except AssertionError as e:
                    print(f"      ✗ Assertion {idx} FAILED: {self._truncate(assertion, 65)}")
                    raise AssertionError(f"Assertion failed: {assertion}") from e
                except Exception as e:
                    print(f"      ⚠️  Assertion {idx} error: {self._truncate(assertion, 65)}")
                    raise RuntimeError(f"Failed to evaluate assertion '{assertion}': {e}") from e

    def _get_debug_suggestion(self, failed_step: Dict[str, Any], error_message: str) -> str:
        system_prompt = """You are an expert Playwright test automation debugger.
Analyze the failed test step and error, provide concise actionable suggestions.
Format as a numbered list of 2-4 specific suggestions."""

        user_prompt = f"""Failed Step:
Action: {failed_step.get('action')}
Target: {failed_step.get('target')}
Value: {failed_step.get('value')}
Assertions: {failed_step.get('assertions', [])}
Error: {error_message}
Provide debugging suggestions:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            content = response.content
            if isinstance(content, list):
                if len(content) > 0 and isinstance(content[0], dict) and 'text' in content[0]:
                    return content[0]['text']
            return str(content)
        except Exception as e:
            return f"Could not generate debug suggestion: {e}"

    def _format_step_description(self, step: Dict[str, Any]) -> str:
        action = step.get("action", "unknown")
        target = step.get("target", "")
        value = step.get("value", "")
        if "params" in step:
            params = step["params"]
            target = params.get("target", target) or params.get("url", target) or params.get("selector", target)
            value = params.get("value", value)
        descriptions = {
            "open":   f"Open '{target}'",
            "click":  f"Click '{target}'",
            "fill":   f"Fill '{target}' with '{value}'",
            "wait":   f"Wait {value}ms",
            "select": f"Select '{value}' from '{target}'",
            "hover":  f"Hover over '{target}'",
            "scroll": f"Scroll to '{target}'",
            "clear":  f"Clear field '{target}'",
        }
        return descriptions.get(action, f"{action.capitalize()} on '{target}'")

    def _truncate(self, text: str, max_length: int) -> str:
        return text if len(text) <= max_length else text[:max_length - 3] + "..."

    def _print_header(self, test_name: str, total_steps: int):
        print("\n" + "="*70)
        print(f"🧪 TEST EXECUTION : {test_name}")
        print(f"📊 Total Steps    : {total_steps}")
        print(f"⏰ Started        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

    def _print_debug_suggestion(self, suggestion: str):
        print("┌" + "─"*68 + "┐")
        print("│ 🤖 AI DEBUG SUGGESTIONS" + " "*44 + "│")
        print("├" + "─"*68 + "┤")
        for line in suggestion.split('\n'):
            if line.strip():
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
        print("\n" + "="*70)
        print("📋 TEST EXECUTION SUMMARY")
        print("="*70)
        status_icon = "✅" if results["status"] == "PASS" else "❌"
        print(f"{status_icon} Status          : {results['status']}")
        print(f"📊 Steps Executed : {results['steps_executed']}/{results['total_steps']}")
        print(f"✅ Passed         : {results['steps_passed']}")
        print(f"❌ Failed         : {results['steps_failed']}")
        print(f"📈 Pass Rate      : {(results['steps_passed'] / results['total_steps']) * 100:.1f}%")
        print(f"⏱️  Execution Time : {results['execution_time']}")
        if results["error"]:
            print(f"\n❌ Error at Step {results['failed_step']}:")
            print(f"   {results['error']}")
        print("="*70 + "\n")


if __name__ == "__main__":

    test_steps = [

        # 1. Open site
        {"action": "open",   "target": "https://www.saucedemo.com/",
         "assertions": ["expect(page).to_have_title(re.compile('Swag Labs'))"]},

        # 2. Fill username
        {"action": "fill",   "params": {"selector": "#user-name", "value": "standard_user"},
         "assertions": []},

        # 3. Fill password
        {"action": "fill",   "params": {"selector": "#password", "value": "secret_sauce"},
         "assertions": []},

        # 4. Login
        {"action": "click",  "target": "#login-button",
         "assertions": [
             "expect(page).to_have_url(re.compile('.*inventory.*'))",
             "expect(page.locator('.inventory_item')).to_have_count(6)"
         ]},

        # 5. Sort low to high
        {"action": "select", "params": {"selector": ".product_sort_container", "value": "lohi"},
         "assertions": ["expect(page.locator('.product_sort_container')).to_have_value('lohi')"]},

        # 6. Wait for sort to apply
        {"action": "wait",   "value": "1000",
         "assertions": []},

        # 7. Hover over first product
        {"action": "hover",  "target": ".inventory_item:first-child .inventory_item_img",
         "assertions": []},

        # 8. Click first product to view detail
        {"action": "click",  "target": ".inventory_item:first-child .inventory_item_name",
         "assertions": [
             "expect(page).to_have_url(re.compile('.*inventory-item.*'))",
             "expect(page.locator('.inventory_details_name')).to_be_visible()",
             "expect(page.locator('.inventory_details_price')).to_be_visible()"
         ]},

        # 9. Add to cart from detail page
        {"action": "click",  "target": "button[id='add-to-cart']",
         "assertions": ["expect(page.locator('.shopping_cart_badge')).to_have_text('1')"]},

        # 10. Go back to inventory
        {"action": "click",  "target": "#back-to-products",
         "assertions": ["expect(page).to_have_url(re.compile('.*inventory.*'))"]},

        # 11. Wait for inventory to load
        {"action": "wait",   "value": "1000",
         "assertions": []},

        # 12. Add backpack to cart
        {"action": "click",  "target": "#add-to-cart-sauce-labs-backpack",
         "assertions": ["expect(page.locator('.shopping_cart_badge')).to_have_text('2')"]},

        # 13. Open cart
        {"action": "click",  "target": ".shopping_cart_link",
         "assertions": [
             "expect(page).to_have_url(re.compile('.*cart.*'))",
             "expect(page.locator('.cart_item')).to_have_count(2)"
         ]},

        # 14. Remove first item from cart
        {"action": "click",  "target": ".cart_item:first-child button",
         "assertions": ["expect(page.locator('.cart_item')).to_have_count(1)"]},

        # 15. Proceed to checkout
        {"action": "click",  "target": "#checkout",
         "assertions": ["expect(page).to_have_url(re.compile('.*checkout-step-one.*'))"]},

        # 16. Fill checkout info
        {"action": "fill",   "params": {"selector": "#first-name", "value": "John"},  "assertions": []},
        {"action": "fill",   "params": {"selector": "#last-name",  "value": "Doe"},   "assertions": []},
        {"action": "fill",   "params": {"selector": "#postal-code","value": "10001"}, "assertions": []},

        # 17. Continue to order overview
        {"action": "click",  "target": "#continue",
         "assertions": [
             "expect(page).to_have_url(re.compile('.*checkout-step-two.*'))",
             "expect(page.locator('.summary_info')).to_be_visible()"
         ]},

        # 18. Scroll to see total price
        {"action": "scroll", "target": ".summary_total_label",
         "assertions": []},

        # 19. Finish order
        {"action": "click",  "target": "#finish",
         "assertions": [
             "expect(page).to_have_url(re.compile('.*checkout-complete.*'))",
             "expect(page.locator('.complete-header')).to_contain_text('Thank you')"
         ]},

        # 20. Logout
        {"action": "click",  "target": "#react-burger-menu-btn",
         "assertions": ["expect(page.locator('#logout_sidebar_link')).to_be_visible()"]},
        {"action": "click",  "target": "#logout_sidebar_link",
         "assertions": [
             "expect(page.locator('#login-button')).to_be_visible()"
         ]},
    ]

    try:
        executor = PlaywrightExecutor(headless=False, use_llm_debugging=True)
        result = executor.execute_test(
            test_steps,
            test_name="Sauce Demo — Full E-Commerce Journey"
        )
    except Exception as e:
        print(f"💥 Critical error: {e}")
