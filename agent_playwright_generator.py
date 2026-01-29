import os
import json
from typing import List, Dict, Any, TypedDict, Union

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ==================================================
# ENV SETUP
# ==================================================

load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment")

# ==================================================
# MODEL SETUP
# ==================================================

REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=REPO_ID,
    temperature=0.2,          # lower = better JSON
    max_new_tokens=800,
)

chat_model = ChatHuggingFace(llm=llm)

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

    response = chat_model.invoke(messages)
    content = response.content

    if isinstance(content, dict) and "text" in content:
        content = content["text"]

    if not isinstance(content, str):
        raise ValueError("LLM did not return text")

    cleaned = clean_json_output(content)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM:\n{cleaned}") from e

    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list")

    return parsed


# ==================================================
# PLAYWRIGHT JS GENERATOR
# ==================================================

def generate_playwright_js(commands: List[Dict[str, Any]]) -> str:
    lines = [
        "const { test, expect } = require('@playwright/test');",
        "",
        "test('Generated Test', async ({ page }) => {",
    ]

    for cmd in commands:
        action = cmd.get("action")
        params = cmd.get("params", {})

        if action == "open":
            lines.append(f"  await page.goto('{params['url']}');")

        elif action == "fill":
            lines.append(
                f"  await page.fill('{params['selector']}', '{params['value']}');"
            )

        elif action == "click":
            lines.append(
                f"  await page.click('{params['selector']}');"
            )

        elif action == "assert":
            lines.append(
                f"  await expect(page.locator('{params['selector']}')).toHaveText('{params['value']}');"
            )

    lines.append("});")
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
    commands = parse_with_llm(state["user_input"])
    return {
        **state,
        "parsed_commands": commands,
    }


def generate_node(state: AgentState) -> AgentState:
    code = generate_playwright_js(state["parsed_commands"])
    return {
        **state,
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
    Open the login page at https://practice.automationtesting.in/.
    Enter username as user1 in #username.
    Enter password as pass123 in #password.
    Click the login button #login.
    """

    result = app.invoke(
        {
            "user_input": test_case,
            "parsed_commands": [],
            "generated_code": "",
        }
    )

    print("\n================ PARSED COMMANDS ================\n")
    print(json.dumps(result["parsed_commands"], indent=2))

    print("\n============= GENERATED PLAYWRIGHT ==============\n")
    print(result["generated_code"])
