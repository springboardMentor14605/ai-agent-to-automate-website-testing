"""
Playwright Agent — LangGraph-based test automation pipeline.

Pipeline:  scout → parse → enrich → generate → execute

The 'scout' node navigates to the target URL first and extracts
the actual page structure (form fields, buttons, selectors).
This lets the 'parse' node provide the LLM with real selectors
instead of guessing.

All graph nodes are synchronous. The 'execute' node calls
PlaywrightExecutor.execute_test() which spawns a subprocess
(avoiding any asyncio / Windows event-loop issues).
"""
import os
import json
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import project modules
from llm_assertion_generator import LLMAssertionGenerator
from assertion_generator import AssertionGenerator
from playwright_executor import PlaywrightExecutor
from page_scout import scout_page

# ==================================================
# ENV SETUP
# ==================================================

load_dotenv()

api_key = os.getenv("ABHAY_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY (or ABHAY_API_KEY) not found in environment")

# ==================================================
# MODEL SETUP
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Initialize components
assertion_gen_rule = AssertionGenerator()
assertion_gen_llm = LLMAssertionGenerator(api_key=os.environ["GOOGLE_API_KEY"])
executor = PlaywrightExecutor(headless=True, slow_mo=500)

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are an instruction parser for an automated web testing system.
Convert the given natural language test case into structured test commands.

Output ONLY valid JSON.
No markdown.
No comments.
No explanation.

Allowed actions:
- open: { "target": "url", "value": "" }
- fill: { "target": "selector", "value": "text to type" }
- click: { "target": "selector", "value": "" }
- wait: { "target": "", "value": "milliseconds" }

Parameters:
- action: One of the allowed actions.
- target: The selector or URL.  Use EXACT CSS selectors from the page structure provided.
- value: The input text or wait time.
- expected: (Optional) Expected result description.

IMPORTANT RULES:
- Use the EXACT selectors from the page structure if provided.
- Do NOT invent selectors. Only use selectors that appear in the page elements list.
- Match fields by their "role" (username, password, email, phone, submit).
- For the submit/login button, use the selector from the element with role "submit".

Output format:
[
  {
    "action": "<action>",
    "target": "<target>",
    "value": "<value>",
    "expected": "<optional expectation string>"
  }
]
"""

# ==================================================
# HELPERS
# ==================================================

def clean_json_output(text: str) -> str:
    """Remove accidental markdown/code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
    return text.strip()


def parse_with_llm(instruction: str) -> List[Dict[str, Any]]:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]

    response = llm.invoke(messages)
    content = response.content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            else:
                text_parts.append(str(block))
        content = "".join(text_parts)
    elif isinstance(content, dict) and "text" in content:
        content = content["text"]

    cleaned = clean_json_output(str(content))

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {cleaned}")
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1:
            try:
                parsed = json.loads(cleaned[start:end+1])
            except Exception:
                raise ValueError(f"Invalid JSON from LLM: {e}")
        else:
            raise ValueError(f"Invalid JSON from LLM: {e}")

    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list")

    return parsed


def build_smart_instruction(user_input: str, scout_data: dict) -> str:
    """
    Combine the user's natural-language request with the actual
    page structure discovered by the scout, so the LLM uses real selectors.
    """
    if not scout_data or not scout_data.get("success"):
        # Fallback: just pass the original instruction
        return user_input

    elements = scout_data.get("elements", [])
    if not elements:
        return user_input

    # Build a structured description of the page elements
    element_descriptions = []
    for el in elements:
        role = el.get("role", "unknown")
        selector = el.get("selector", "")
        attrs = el.get("attributes", {})
        text = el.get("text", "")
        in_iframe = el.get("in_iframe", False)

        desc_parts = [f"  - Role: {role}, Selector: {selector}"]
        if text:
            desc_parts.append(f"    Text: {text}")
        if attrs.get("placeholder"):
            desc_parts.append(f"    Placeholder: {attrs['placeholder']}")
        if attrs.get("name"):
            desc_parts.append(f"    Name: {attrs['name']}")
        if in_iframe:
            desc_parts.append(f"    (Inside iframe: {el.get('frame_url', '')})")

        element_descriptions.append("\n".join(desc_parts))

    elements_block = "\n".join(element_descriptions)

    enhanced = f"""{user_input}

--- PAGE STRUCTURE (discovered by scouting the actual page) ---
Page title: {scout_data.get('page_title', '')}
Final URL: {scout_data.get('final_url', '')}

Available form elements on the page:
{elements_block}

IMPORTANT: Use the EXACT selectors listed above. Do NOT use generic selectors like #user-name unless they appear in the list above.
"""
    return enhanced

# ==================================================
# PLAYWRIGHT CODE GENERATOR  (for display / export)
# ==================================================

def generate_playwright_python(commands: List[Dict[str, Any]]) -> str:
    lines = [
        "from playwright.sync_api import sync_playwright",
        "",
        "def run_test():",
        "    with sync_playwright() as p:",
        "        browser = p.chromium.launch(headless=False)",
        "        page = browser.new_page()",
        ""
    ]

    for cmd in commands:
        action = cmd.get("action")
        target = cmd.get("target")
        value = cmd.get("value")

        if action == "open":
            lines.append(f"        page.goto('{target}')")
        elif action == "fill":
            lines.append(f"        page.fill('{target}', '{value}')")
        elif action == "click":
            lines.append(f"        page.click('{target}')")
        elif action == "wait":
            lines.append(f"        page.wait_for_timeout({int(value)})")

    lines.append("")
    lines.append("        browser.close()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    run_test()")

    return "\n".join(lines)

# ==================================================
# LANGGRAPH STATE
# ==================================================

class AgentState(TypedDict):
    user_input: str
    scout_data: Dict[str, Any]
    parsed_commands: List[Dict[str, Any]]
    generated_code: str
    execution_results: Dict[str, Any]

# ==================================================
# LANGGRAPH NODES  (all synchronous)
# ==================================================

def scout_node(state: AgentState) -> AgentState:
    """Navigate to the target URL and extract form elements."""
    print("-> Scouting target page for form elements...")
    scout_data = {}

    # Extract the URL from user_input
    url = ""
    for word in state["user_input"].split():
        if word.startswith("http://") or word.startswith("https://"):
            # Clean trailing punctuation
            url = word.rstrip(".,;!?")
            break

    if url:
        print(f"   Scouting URL: {url}")
        scout_data = scout_page(url)
        if scout_data.get("success"):
            n_elements = len(scout_data.get("elements", []))
            print(f"   Found {n_elements} form element(s) on the page")
            for el in scout_data.get("elements", []):
                print(f"     - {el.get('role', '?'):10s} -> {el.get('selector', '?')}")
        else:
            print(f"   Scout warning: {scout_data.get('error', 'No elements found')}")
    else:
        print("   Could not extract URL from user input")

    return {
        "user_input": state["user_input"],
        "scout_data": scout_data,
        "parsed_commands": [],
        "generated_code": "",
        "execution_results": {}
    }


def parse_node(state: AgentState) -> AgentState:
    """Parse user input + scout data into structured commands via LLM."""
    print(f"-> Parsing user input with page context...")
    try:
        # Enhance the instruction with actual page element info
        enhanced_input = build_smart_instruction(
            state["user_input"],
            state.get("scout_data", {})
        )
        commands = parse_with_llm(enhanced_input)
    except Exception as e:
        print(f"   Parsing error: {e}")
        commands = []

    return {
        "user_input": state["user_input"],
        "scout_data": state.get("scout_data", {}),
        "parsed_commands": commands,
        "generated_code": "",
        "execution_results": {}
    }


def enrich_node(state: AgentState) -> AgentState:
    print("-> Enriching commands with assertions...")
    commands = state["parsed_commands"]
    enriched = []

    for cmd in commands:
        new_cmd = cmd.copy()
        assertions = []

        try:
            assertions = assertion_gen_rule.generate_assertions(new_cmd)
        except Exception as e:
            print(f"   Rule-Based Assertion Error: {e}")

        if not assertions and new_cmd.get("expected"):
            print(f"   Using LLM for expectation: {new_cmd.get('expected')}")
            try:
                assertions = assertion_gen_llm.generate_assertions(new_cmd)
            except Exception as e:
                print(f"   LLM Assertion Error: {e}")

        new_cmd["assertions"] = assertions
        enriched.append(new_cmd)

    return {
        "user_input": state["user_input"],
        "scout_data": state.get("scout_data", {}),
        "parsed_commands": enriched,
        "generated_code": "",
        "execution_results": {}
    }


def generate_node(state: AgentState) -> AgentState:
    print("-> Generating Playwright code...")
    code = generate_playwright_python(state["parsed_commands"])
    return {
        "user_input": state["user_input"],
        "scout_data": state.get("scout_data", {}),
        "parsed_commands": state["parsed_commands"],
        "generated_code": code,
        "execution_results": {}
    }


def execute_node(state: AgentState) -> AgentState:
    """Run Playwright via subprocess (no asyncio conflict)."""
    print("-> Executing test in browser (subprocess)...")
    results = {}
    try:
        # Extract the login URL from the parsed commands for verification
        login_url = ""
        for cmd in state["parsed_commands"]:
            if cmd.get("action") == "open":
                login_url = cmd.get("target", "")
                break

        results = executor.execute_test(state["parsed_commands"], login_url=login_url, is_login_test=True)
        print(f"   Execution Status: {results.get('status', 'UNKNOWN')}")
        if results.get("error"):
            print(f"   Error: {results['error']}")
    except Exception as e:
        print(f"   Execution Layer Error: {e}")
        results = {"status": "FAIL", "error": str(e), "details": [f"[FAIL] {e}"]}

    return {
        "user_input": state["user_input"],
        "scout_data": state.get("scout_data", {}),
        "parsed_commands": state["parsed_commands"],
        "generated_code": state["generated_code"],
        "execution_results": results
    }

# ==================================================
# GRAPH DEFINITION
# ==================================================

graph = StateGraph(AgentState)

graph.add_node("scout", scout_node)
graph.add_node("parse", parse_node)
graph.add_node("enrich", enrich_node)
graph.add_node("generate", generate_node)
graph.add_node("execute", execute_node)

graph.set_entry_point("scout")
graph.add_edge("scout", "parse")
graph.add_edge("parse", "enrich")
graph.add_edge("enrich", "generate")
graph.add_edge("generate", "execute")
graph.add_edge("execute", END)

app = graph.compile()

# ==================================================
# MAIN  (standalone usage)
# ==================================================

if __name__ == "__main__":
    test_case = """
    Open the login page at https://www.saucedemo.com/.
    Enter username as standard_user.
    Enter password as secret_sauce.
    Click the login button.
    """

    print("Starting Agent...")
    try:
        result = app.invoke({
            "user_input": test_case,
            "scout_data": {},
            "parsed_commands": [],
            "generated_code": "",
            "execution_results": {}
        })

        print("\n============= GENERATED PLAYWRIGHT CODE ==============\n")
        print(result["generated_code"])

        print("\n============= EXECUTION RESULTS ==============\n")
        print(json.dumps(result["execution_results"], indent=2))

        output_file = "generated_test_script.py"
        with open(output_file, "w") as f:
            f.write(result["generated_code"])
        print(f"\n[INFO] Generated code saved to {output_file}")

    except Exception as e:
        print(f"\n[ERROR] execution failed: {e}")
