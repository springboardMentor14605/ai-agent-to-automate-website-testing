"""
Execution Module
Runs generated Playwright test steps in a headless/headed Chromium browser.
Captures runtime logs and handles per-step exceptions with AI debug assistance.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from playwright.sync_api import sync_playwright, expect
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


class PlaywrightExecutor:
    """Executes Playwright test steps and returns structured results."""

    def __init__(self, headless: bool = True, use_llm_debugging: bool = False,
                 timeout: int = 15000):
        self.headless = headless
        self.use_llm_debugging = use_llm_debugging
        self.timeout = timeout
        self.llm: Optional[ChatGoogleGenerativeAI] = None

        if use_llm_debugging:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview",
                    temperature=0.3,
                    google_api_key=api_key
                )
            else:
                print("[Executor] Warning: GOOGLE_API_KEY not found – LLM debugging disabled.")
                self.use_llm_debugging = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def execute(self, steps: List[Dict[str, Any]], test_name: str = "Test Suite") -> Dict[str, Any]:
        """
        Execute a list of test steps and return a result dictionary.

        Args:
            steps:     List of step dicts produced by InstructionParser / AssertionGenerator.
            test_name: Label shown in the execution report.

        Returns:
            dict with keys: test_name, status, steps_executed, steps_passed,
            steps_failed, total_steps, error, failed_step, step_results,
            execution_time, timestamp.
        """
        result = self._init_result(test_name, len(steps))
        start = datetime.now()

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(self.timeout)

            try:
                for idx, step in enumerate(steps, 1):
                    step_result = self._run_step(page, step, idx, len(steps))
                    result["step_results"].append(step_result)
                    result["steps_executed"] += 1

                    if step_result["status"] == "PASS":
                        result["steps_passed"] += 1
                    else:
                        result["steps_failed"] += 1
                        result["failed_step"] = idx
                        result["status"] = "FAIL"
                        result["error"] = step_result["error"]

                        if self.use_llm_debugging:
                            result["debug_suggestion"] = self._debug(step, step_result["error"])

                        # Stop on first failure
                        break

            except Exception as e:
                result["status"] = "FAIL"
                result["error"] = str(e)
            finally:
                browser.close()

        result["execution_time"] = str(datetime.now() - start)
        result["timestamp"] = start.strftime("%Y-%m-%d %H:%M:%S")
        return result

    # ------------------------------------------------------------------ #
    # Step dispatch
    # ------------------------------------------------------------------ #

    def _run_step(self, page, step: Dict, idx: int, total: int) -> Dict:
        """Execute a single step and return its result dict."""
        action = step.get("action", "").lower()
        target = self._resolve(step, "target")
        value = self._resolve(step, "value")
        assertions = step.get("assertions", [])
        description = step.get("description", f"Step {idx}")

        step_result = {
            "step_number": idx,
            "action": action,
            "target": target,
            "value": value,
            "description": description,
            "status": "PASS",
            "error": None,
            "assertions_run": 0,
            "assertions_passed": 0,
            "assertions_failed": 0,
            "assertion_details": []
        }

        try:
            # --- Action dispatch ---
            if action == "open":
                url = target if target.startswith("http") else f"https://{target}"
                page.goto(url)
                page.wait_for_load_state("networkidle")

            elif action == "fill":
                page.wait_for_selector(target, state="visible", timeout=self.timeout)
                page.fill(target, value or "")

            elif action == "click":
                page.wait_for_selector(target, state="visible", timeout=self.timeout)
                page.click(target)
                page.wait_for_load_state("domcontentloaded")

            elif action == "select":
                page.wait_for_selector(target, state="visible", timeout=self.timeout)
                page.select_option(target, value or "")

            elif action == "hover":
                page.wait_for_selector(target, state="visible", timeout=self.timeout)
                page.hover(target)

            elif action == "scroll":
                page.locator(target).scroll_into_view_if_needed()

            elif action == "wait":
                ms = int(value) if value else 1000
                page.wait_for_timeout(ms)

            elif action == "clear":
                page.wait_for_selector(target, state="visible", timeout=self.timeout)
                page.fill(target, "")

            elif action == "assert":
                pass  # assertions handled below

            else:
                raise ValueError(f"Unknown action: '{action}'")

            # --- Assertion evaluation ---
            for assertion in assertions:
                assertion = assertion.strip()
                if not assertion.startswith("expect("):
                    continue

                step_result["assertions_run"] += 1
                detail = {"assertion": assertion, "status": "PASS", "error": None}

                try:
                    eval(assertion, {"expect": expect, "page": page, "re": re})
                    step_result["assertions_passed"] += 1
                except AssertionError as ae:
                    detail["status"] = "FAIL"
                    detail["error"] = str(ae)
                    step_result["assertions_failed"] += 1
                    step_result["status"] = "FAIL"
                    step_result["error"] = f"Assertion failed: {assertion}"
                except Exception as ae:
                    detail["status"] = "ERROR"
                    detail["error"] = str(ae)
                    step_result["assertions_failed"] += 1
                    step_result["status"] = "FAIL"
                    step_result["error"] = f"Assertion error: {assertion} → {ae}"

                step_result["assertion_details"].append(detail)

                if step_result["status"] == "FAIL":
                    break  # stop on first assertion failure

        except Exception as e:
            step_result["status"] = "FAIL"
            step_result["error"] = str(e)

        return step_result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve(self, step: Dict, key: str) -> str:
        """Return value from step or nested params dict."""
        val = step.get(key, "")
        if not val and "params" in step:
            params = step["params"]
            val = (params.get(key) or params.get("url") or
                   params.get("selector") or "") if key == "target" else params.get(key, "")
        return str(val) if val else ""

    def _debug(self, step: Dict, error: str) -> str:
        """Ask the LLM for debugging suggestions for a failed step."""
        if not self.llm:
            return ""
        system = ("You are an expert Playwright debugger. Given a failed test step and its error, "
                  "provide 2-4 numbered, specific, actionable suggestions to fix it.")
        user = (f"Failed Step:\n  Action: {step.get('action')}\n  Target: {step.get('target')}\n"
                f"  Value: {step.get('value')}\n  Assertions: {step.get('assertions', [])}\n"
                f"Error: {error}\n\nDebugging suggestions:")
        try:
            resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            content = resp.content
            if isinstance(content, list):
                return content[0].get("text", "") if content else ""
            return str(content)
        except Exception as e:
            return f"Could not generate suggestion: {e}"

    @staticmethod
    def _init_result(test_name: str, total: int) -> Dict:
        return {
            "test_name": test_name,
            "status": "PASS",
            "steps_executed": 0,
            "steps_passed": 0,
            "steps_failed": 0,
            "total_steps": total,
            "failed_step": None,
            "error": None,
            "debug_suggestion": None,
            "step_results": [],
            "execution_time": None,
            "timestamp": None
        }
