import os
import json
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

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
- open: { "url": string }
- fill: { "selector": string, "value": string }
- click: { "selector": string }
- assert: { "selector": string, "value": string }

Output format:
[
  {
    "action": "<action>",
    "params": { ... }
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

    # Handle Gemini sometimes returning a list of parts or dict
    if isinstance(content, list):
         # Extract text from blocks if possible
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
        params = cmd.get("params", {})

        if action == "open":
            lines.append(f"        page.goto('{params['url']}')")

        elif action == "fill":
            lines.append(
                f"        page.fill('{params['selector']}', '{params['value']}')"
            )

        elif action == "click":
            lines.append(
                f"        page.click('{params['selector']}')"
            )

        elif action == "assert":
            lines.append(
                f"        expect(page.locator('{params['selector']}')).to_have_text('{params['value']}')"
            )

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

# ==================================================
# LANGGRAPH NODES
# ==================================================

def parse_node(state: AgentState) -> AgentState:
    print(f"-> Parsing user input: {state['user_input'][:50]}...")
    commands = parse_with_llm(state["user_input"])
    return {
        "user_input": state["user_input"],
        "parsed_commands": commands,
        "generated_code": state.get("generated_code", "")
    }

def generate_node(state: AgentState) -> AgentState:
    print("-> Generating Playwright code...")
    code = generate_playwright_python(state["parsed_commands"])
    return {
        "user_input": state["user_input"],
        "parsed_commands": state["parsed_commands"],
        "generated_code": code,
    }

# ==================================================
# GRAPH DEFINITION
# ==================================================

graph = StateGraph(AgentState)

graph.add_node("parse", parse_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("parse")
graph.add_edge("parse", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    test_case = """
    Open the login page at https://practice.automationtesting.in/my-account/.
    Enter username as kabir@google.com in #username.
    Enter password as pass123 in #password.
    Click the login button input[name="login"].
    """

    print("Starting Agent...")
    try:
        result = app.invoke(
            {
                "user_input": test_case,
                "parsed_commands": [],
                "generated_code": "",
            }
        )

        print("\n================ PARSED COMMANDS ================\n")
        print(json.dumps(result["parsed_commands"], indent=2))

        print("\n============= GENERATED PLAYWRIGHT CODE ==============\n")
        print(result["generated_code"])
        
        # Optional: Save to file to run it
        output_file = "generated_test_script.py"
        with open(output_file, "w") as f:
            f.write(result["generated_code"])
        print(f"\n[INFO] Generated code saved to {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] execution failed: {e}")
