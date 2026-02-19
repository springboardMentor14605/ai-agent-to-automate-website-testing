"""
FastAPI backend for the Automated Website Testing GUI.

Provides a REST API that accepts login credentials and a target URL,
runs a Playwright-based test via a LangGraph agent, and returns
status, screenshot URL, generated code, and step-by-step details.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys
from dotenv import load_dotenv

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load env variables
load_dotenv()

# Import the agent application
try:
    from playwright_agent import app as agent_app
except ImportError as e:
    print(f"Error importing playwright_agent: {e}")
    agent_app = None

app = FastAPI(
    title="Automated Website Testing",
    description="AI-powered website login testing with Playwright",
    version="3.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount screenshots directory to serve images
screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)
app.mount("/screenshots", StaticFiles(directory=screenshots_dir), name="screenshots")


class TestRequest(BaseModel):
    username: str
    password: str
    phone: Optional[str] = None
    url: str


class InstructionRequest(BaseModel):
    instruction: str
    url: str


@app.post("/api/run-test")
def run_test(req: TestRequest):
    """
    Run a login test against the given URL.

    The agent pipeline now auto-discovers form selectors:
      1. Scout — navigates to the URL and extracts form elements
      2. Parse — LLM uses real selectors to build test steps
      3. Enrich — adds assertions
      4. Generate — produces displayable Playwright code
      5. Execute — runs the test in a subprocess
    """
    if not agent_app:
        raise HTTPException(
            status_code=500,
            detail="Agent application not loaded. Check server logs."
        )

    # Build GENERIC instruction — no hardcoded selectors!
    # The scout node will discover the real selectors from the page.
    user_input = f"""
    Open the login page at {req.url}.
    Enter username as {req.username} in the username field.
    Enter password as {req.password} in the password field.
    Click the login/sign-in button.
    """

    try:
        result = agent_app.invoke({
            "user_input": user_input,
            "scout_data": {},
            "parsed_commands": [],
            "generated_code": "",
            "execution_results": {}
        })

        exec_results = result.get("execution_results", {})
        status = exec_results.get("status", "UNKNOWN")
        error = exec_results.get("error")
        screenshot = exec_results.get("screenshot")
        details = exec_results.get("details", [])

        # Convert screenshot path to a URL the frontend can fetch
        screenshot_url = None
        if screenshot and os.path.isfile(screenshot):
            filename = os.path.basename(screenshot)
            screenshot_url = f"http://localhost:8000/screenshots/{filename}"

        return {
            "status": status,
            "error": error,
            "screenshot_url": screenshot_url,
            "generated_code": result.get("generated_code"),
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-instruction")
def run_instruction(req: InstructionRequest):
    """
    Execute a natural language instruction against the given URL.

    Pipeline:
      1. Validate the instruction
      2. Scout the target page for elements
      3. Parse the instruction using LLM (with page context)
      4. Execute the parsed steps via Playwright subprocess
    """
    from instruction_processor import (
        parse_instruction, validate_instruction, EXAMPLE_COMMANDS
    )
    from page_scout import scout_page
    from playwright_executor import PlaywrightExecutor

    # 1. Validate
    validation = validate_instruction(req.instruction)
    if not validation["is_valid"]:
        return {
            "status": "FAIL",
            "error": validation["error"],
            "intent": "error",
            "description": "Validation failed",
            "entities": {},
            "steps_parsed": [],
            "details": [f"[FAIL] {validation['error']}"],
            "screenshot_url": None,
            "generated_code": None,
        }

    # 2. Scout the target page
    print(f"-> Scouting page: {req.url}")
    scout_data = scout_page(req.url)
    if scout_data.get("success"):
        n = len(scout_data.get("elements", []))
        print(f"   Found {n} element(s) on page")
    else:
        print(f"   Scout warning: {scout_data.get('error', 'No elements found')}")

    # 3. Parse instruction with LLM
    print(f"-> Parsing instruction: {req.instruction}")
    parsed = parse_instruction(req.instruction, req.url, scout_data)

    if not parsed.get("success"):
        return {
            "status": "FAIL",
            "error": parsed.get("error", "Failed to parse instruction"),
            "intent": parsed.get("intent", "error"),
            "description": parsed.get("description", ""),
            "entities": parsed.get("entities", {}),
            "steps_parsed": [],
            "details": [f"[FAIL] {parsed.get('error', 'Parse error')}"],
            "screenshot_url": None,
            "generated_code": None,
        }

    steps = parsed.get("steps", [])
    print(f"   Intent: {parsed.get('intent')}")
    print(f"   Steps: {len(steps)}")

    # Ensure the first step opens the URL if not already included
    if steps and steps[0].get("action") != "open":
        steps.insert(0, {
            "action": "open",
            "target": req.url,
            "value": "",
            "description": f"Navigate to {req.url}"
        })

    # 4. Execute via Playwright subprocess
    print("-> Executing instruction steps in browser...")
    executor = PlaywrightExecutor(headless=True, slow_mo=500)

    # Convert steps to executor format (action/target/value)
    exec_steps = []
    for s in steps:
        exec_steps.append({
            "action": s.get("action", ""),
            "target": s.get("target", ""),
            "value": s.get("value", ""),
        })

    exec_results = executor.execute_test(exec_steps, login_url=req.url, is_login_test=False)

    # Convert screenshot path to URL
    screenshot_url = None
    screenshot = exec_results.get("screenshot")
    if screenshot and os.path.isfile(screenshot):
        filename = os.path.basename(screenshot)
        screenshot_url = f"http://localhost:8000/screenshots/{filename}"

    # Build generated code from steps
    code_lines = [
        "from playwright.sync_api import sync_playwright",
        "",
        "def run_instruction():",
        "    with sync_playwright() as p:",
        "        browser = p.chromium.launch(headless=False)",
        "        page = browser.new_page()",
        ""
    ]
    for s in steps:
        action = s.get("action")
        target = s.get("target", "")
        value = s.get("value", "")
        desc = s.get("description", "")
        if desc:
            code_lines.append(f"        # {desc}")
        if action == "open":
            code_lines.append(f"        page.goto('{target}')")
        elif action == "fill":
            code_lines.append(f"        page.fill('{target}', '{value}')")
        elif action == "click":
            code_lines.append(f"        page.click('{target}')")
        elif action == "wait":
            code_lines.append(f"        page.wait_for_timeout({int(value)})")
        elif action == "select":
            code_lines.append(f"        page.select_option('{target}', '{value}')")
        elif action == "scroll":
            direction = "-300" if value == "up" else "300"
            if target:
                code_lines.append(f"        page.locator('{target}').evaluate('el => el.scrollBy(0, {direction})')")
            else:
                code_lines.append(f"        page.evaluate('window.scrollBy(0, {direction})')")
        elif action == "press":
            code_lines.append(f"        page.keyboard.press('{value or 'Enter'}')")
        code_lines.append("")
    code_lines.append("        browser.close()")
    code_lines.append("")
    code_lines.append("if __name__ == '__main__':")
    code_lines.append("    run_instruction()")
    generated_code = "\n".join(code_lines)

    return {
        "status": exec_results.get("status", "UNKNOWN"),
        "error": exec_results.get("error"),
        "intent": parsed.get("intent", "custom"),
        "description": parsed.get("description", ""),
        "entities": parsed.get("entities", {}),
        "steps_parsed": [
            s.get("description", f"{s.get('action')} {s.get('target', '')}")
            for s in steps
        ],
        "details": exec_results.get("details", []),
        "screenshot_url": screenshot_url,
        "generated_code": generated_code,
    }


@app.get("/api/example-commands")
def get_example_commands():
    """Return example commands for the instruction box UI."""
    from instruction_processor import EXAMPLE_COMMANDS
    return {"commands": EXAMPLE_COMMANDS}


@app.get("/api/health")
def health():
    return {"status": "ok", "agent_loaded": agent_app is not None}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["tmp*", "*.tmp", "screenshots/*"],
    )
