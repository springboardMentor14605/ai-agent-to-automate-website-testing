import os
import json
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import new modules
from llm_assertion_generator import LLMAssertionGenerator
from assertion_generator import AssertionGenerator
from playwright_executor import PlaywrightExecutor

# ==================================================
# ENV SETUP
# ==================================================

load_dotenv()

# Ensure API Key is set for Gemini
api_key = os.getenv("ABHAY_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY (or ABHAY_API_KEY) not found in environment")

# ==================================================
# MODEL SETUP
# ==================================================

# Using Gemini 2.5 Flash as identified in available models
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# Initialize Component Classes
assertion_gen_rule = AssertionGenerator()
assertion_gen_llm = LLMAssertionGenerator(api_key=os.environ["GOOGLE_API_KEY"])
executor = PlaywrightExecutor(headless=True) # Default to headless as per Doc 2

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
- target: The selector or URL.
- value: The input text or wait time.
- expected: (Optional) A description of the expected result after this step (e.g., "The login button should be visible", "URL should be /dashboard").

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
    # New prompt structure implies we need to be careful with the output
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]

    response = llm.invoke(messages)
    content = response.content

    # Handle Gemini sometimes returning a list of parts or dict
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
        # Robustness: Try to find JSON array in text
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1:
            try:
                parsed = json.loads(cleaned[start:end+1])
            except:
                raise ValueError(f"Invalid JSON from LLM: {e}")
        else:
             raise ValueError(f"Invalid JSON from LLM: {e}")

    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list")

    return parsed

# ==================================================
# PLAYWRIGHT PYTHON GENERATOR
# ==================================================

def generate_playwright_python(commands: List[Dict[str, Any]]) -> str:
    lines = [
        "from playwright.sync_api import sync_playwright, expect",
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
        assertions = cmd.get("assertions", [])

        if action == "open":
            lines.append(f"        page.goto('{target}')")
        elif action == "fill":
            lines.append(f"        page.fill('{target}', '{value}')")
        elif action == "click":
            lines.append(f"        page.click('{target}')")
        elif action == "wait":
            lines.append(f"        page.wait_for_timeout({int(value)})")
        
        # Add generated assertions
        for assertion in assertions:
            lines.append(f"        {assertion}")

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
    parsed_commands: List[Dict[str, Any]]
    generated_code: str
    execution_results: Dict[str, Any]

# ==================================================
# LANGGRAPH NODES
# ==================================================

def parse_node(state: AgentState) -> AgentState:
    print(f"-> Parsing user input: {state['user_input'][:50]}...")
    try:
        commands = parse_with_llm(state["user_input"])
    except Exception as e:
        print(f"Parsing error: {e}")
        commands = []
        
    return {
        "user_input": state["user_input"],
        "parsed_commands": commands,
        "generated_code": "",
        "execution_results": {}
    }

def enrich_node(state: AgentState) -> AgentState:
    print("-> Enriching commands with assertions...")
    commands = state["parsed_commands"]
    enriched_commands = []
    
    for cmd in commands:
        new_cmd = cmd.copy()
        assertions = []
        
        # 1. Try Rule-Based Generation
        try:
            assertions = assertion_gen_rule.generate_assertions(new_cmd)
        except Exception as e:
            print(f"Rule Based Assertion Error: {e}")

        # 2. If no rule-based assertions and there is an expectation, try LLM
        if not assertions and new_cmd.get("expected"):
            print(f"   Using LLM for expectation: {new_cmd.get('expected')}")
            try:
                assertions = assertion_gen_llm.generate_assertions(new_cmd)
            except Exception as e:
                print(f"LLM Assertion Error: {e}")
            
        new_cmd["assertions"] = assertions
        enriched_commands.append(new_cmd)
        
    return {
        "user_input": state["user_input"],
        "parsed_commands": enriched_commands,
        "generated_code": "",
        "execution_results": {}
    }

def generate_node(state: AgentState) -> AgentState:
    print("-> Generating Playwright code...")
    code = generate_playwright_python(state["parsed_commands"])
    return {
        "user_input": state["user_input"],
        "parsed_commands": state["parsed_commands"],
        "generated_code": code,
        "execution_results": {}
    }

def execute_node(state: AgentState) -> AgentState:
    print("-> Executing test in headless browser...")
    results = {}
    try:
        results = executor.execute_test(state["parsed_commands"])
        print(f"   Execution Status: {results.get('status', 'UNKNOWN')}")
        if results.get('error'):
            print(f"   Error: {results['error']}")
    except Exception as e:
        print(f"   Execution Layer Error: {e}")
        results = {"status": "FAIL", "error": str(e)}

    return {
        "user_input": state["user_input"],
        "parsed_commands": state["parsed_commands"],
        "generated_code": state["generated_code"],
        "execution_results": results
    }

# ==================================================
# GRAPH DEFINITION
# ==================================================

graph = StateGraph(AgentState)

graph.add_node("parse", parse_node)
graph.add_node("enrich", enrich_node)
graph.add_node("generate", generate_node)
graph.add_node("execute", execute_node)

graph.set_entry_point("parse")
graph.add_edge("parse", "enrich")
graph.add_edge("enrich", "generate")
graph.add_edge("generate", "execute")
graph.add_edge("execute", END)

app = graph.compile()

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    # Test Case matching Doc 1 & 3 examples
    test_case = """
    Open the login page at https://practice.automationtesting.in/my-account/ expecting the url to be correct.
    Enter username as kabir@google.com in #username.
    Enter password as pass123 in #password.
    Click the login button input[name="login"] expecting the text 'Hello' to be visible.
    """

    print("Starting Agent...")
    try:
        result = app.invoke(
            {
                "user_input": test_case,
                "parsed_commands": [],
                "generated_code": "",
                "execution_results": {}
            }
        )
        
        print("\n============= GENERATED PLAYWRIGHT CODE ==============\n")
        print(result["generated_code"])
        
        print("\n============= EXECUTION RESULTS ==============\n")
        print(json.dumps(result["execution_results"], indent=2))
        
        # Update generated test script
        output_file = "generated_test_script.py"
        with open(output_file, "w") as f:
            f.write(result["generated_code"])
        print(f"\n[INFO] Generated code saved to {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] execution failed: {e}")
