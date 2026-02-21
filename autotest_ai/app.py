import os
import json
import traceback
from typing import TypedDict, Optional, List
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from llm_assertion_generator import LLMAssertionGenerator, extract_text
from playwright_executor import PlaywrightExecutor

# ── API KEY ── Set your Google Gemini API key here ──────────────
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
# ─────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)


# ─────────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# Shared state passed between every node in the graph
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    url:              str
    raw_instructions: str            # user's plain English instructions
    enriched_steps:   Optional[List] # parsed steps + assertions (from single LLM call)
    step_details:     Optional[List] # UI-friendly copy for reporting
    test_result:      Optional[dict] # Playwright execution output
    report:           Optional[dict] # final report sent to frontend
    error:            Optional[str]


# ─────────────────────────────────────────────────────────────────
# NODE 1 — convert_user_instruction_to_actions
#
# Takes the user's natural language instructions and in a SINGLE
# Gemini call produces:
#   - parsed action steps (open / fill / click / wait)
#   - Playwright Python assertions for each step
# ─────────────────────────────────────────────────────────────────
def convert_user_instruction_to_actions(state: AgentState) -> AgentState:
    print("[Node 1/3] convert_user_instruction_to_actions → 1 LLM call")
    try:
        generator = LLMAssertionGenerator(GEMINI_API_KEY)

        prompt = f"""You are an expert QA automation engineer.

The user wants to test this website: {state["url"]}

Here are their natural language test instructions:
\"\"\"{state["raw_instructions"]}\"\"\"

Do the following in ONE response:
1. Parse the instructions into structured test steps.
2. For each step, also generate the Playwright Python assertion.

Return ONLY a valid JSON array. Each object must have:
  - action   : one of "open", "fill", "click", "wait"
  - target   : CSS selector or full URL
  - value    : text to type (only for "fill" actions, else null)
  - expected : natural language description of what should happen
  - assertion: a single valid Playwright Python expect() string, or null

Rules:
- The FIRST step must always be action="open" with target="{state["url"]}"
- For fill steps, extract the exact value from the instructions
- For click steps, use the best CSS selector (prefer IDs like #submit)
- Assertions must use Python Playwright syntax only:
    expect(page).to_have_url("...")
    expect(page.locator("#id")).to_be_visible()
    expect(page.locator("#id")).to_have_value("text")
    expect(page.get_by_text("text")).to_be_visible()
    expect(page).to_have_title("title")
- If multiple elements match, add .first: expect(page.locator("x").first).to_be_visible()
- Use null for assertion if no meaningful check applies
- NEVER invent selectors — only use what the instructions clearly imply

Return ONLY the JSON array. No markdown, no explanation.

Example output:
[
  {{"action":"open","target":"https://example.com","value":null,"expected":"Login page loads","assertion":"expect(page.locator(\\"h1\\")).to_be_visible()"}},
  {{"action":"fill","target":"#username","value":"student","expected":"Username field shows student","assertion":"expect(page.locator(\\"#username\\")).to_have_value(\\"student\\")"}},
  {{"action":"click","target":"#submit","value":null,"expected":"Form is submitted","assertion":null}},
  {{"action":"wait","target":"page","value":null,"expected":"Success page loads","assertion":"expect(page.get_by_text(\\"Congratulations\\")).to_be_visible()"}}
]"""

        response = generator.llm.invoke([HumanMessage(content=prompt)])
        raw = extract_text(response.content)

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        steps = json.loads(raw)

        enriched_steps = []
        step_details   = []

        for i, step in enumerate(steps):
            assertion  = step.get("assertion")
            assertions = [assertion] if assertion else []
            enriched_steps.append({
                "action":     step.get("action"),
                "target":     step.get("target"),
                "value":      step.get("value"),
                "expected":   step.get("expected"),
                "assertions": assertions
            })
            step_details.append({
                "step":       i + 1,
                "action":     step.get("action"),
                "target":     step.get("target"),
                "value":      step.get("value"),
                "expected":   step.get("expected"),
                "assertions": assertions,
                "status":     "PENDING"
            })

        # Guarantee first step is always "open"
        if not enriched_steps or enriched_steps[0].get("action") != "open":
            open_e = {"action":"open","target":state["url"],"value":None,
                      "expected":"Page should load","assertions":[]}
            open_d = {"step":0,"action":"open","target":state["url"],"value":None,
                      "expected":"Page should load","assertions":[],"status":"PENDING"}
            enriched_steps.insert(0, open_e)
            step_details.insert(0, open_d)
            for i, s in enumerate(step_details):
                s["step"] = i + 1
        else:
            enriched_steps[0]["target"] = state["url"]
            step_details[0]["target"]   = state["url"]

        state["enriched_steps"] = enriched_steps
        state["step_details"]   = step_details
        print(f"         ✓ {len(enriched_steps)} steps parsed + assertions generated (1 LLM call)")

    except json.JSONDecodeError as e:
        state["error"] = f"LLM returned invalid JSON: {str(e)}"
        print(f"         ✗ {state['error']}")
    except Exception as e:
        state["error"] = str(e)
        print(f"         ✗ {state['error']}")

    return state


# ─────────────────────────────────────────────────────────────────
# NODE 2 — execute_test_case
#
# PDF: "Execution Phase"
# Runs all steps in a real headless Chromium browser via Playwright
# Tracks pass/fail per step and captures error messages
# ─────────────────────────────────────────────────────────────────
def execute_test_case(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    print(f"[Node 2/3] execute_test_case → {len(state['enriched_steps'])} steps")
    try:
        executor = PlaywrightExecutor(headless=True)
        result   = executor.execute_test(state["enriched_steps"], state["step_details"])
        state["test_result"] = result
        print(f"         ✓ Execution complete — {result['status']}")
    except Exception as e:
        state["error"] = f"Playwright execution failed: {str(e)}"
        print(f"         ✗ {state['error']}")

    return state


# ─────────────────────────────────────────────────────────────────
# NODE 3 — generate_test_report
#
# PDF: "Reporting Phase"
# Compiles the final human-readable pass/fail report
# Summarizes passed/failed/skipped steps for the UI
# ─────────────────────────────────────────────────────────────────
def generate_test_report(state: AgentState) -> AgentState:
    print("[Node 3/3] generate_test_report → compiling report")

    result       = state.get("test_result", {})
    step_details = state.get("step_details", [])
    steps        = result.get("steps", step_details)

    state["report"] = {
        "url":            state["url"],
        "instructions":   state["raw_instructions"],
        "total_steps":    len(state.get("enriched_steps", [])),
        "steps_executed": result.get("steps_executed", 0),
        "status":         result.get("status", "FAIL"),
        "error":          result.get("error") or state.get("error"),
        "steps":          steps,
        "summary": {
            "passed":  sum(1 for s in steps if s["status"] == "PASS"),
            "failed":  sum(1 for s in steps if s["status"] == "FAIL"),
            "skipped": sum(1 for s in steps if s["status"] == "PENDING"),
        }
    }
    print(f"         ✓ Report ready → {state['report']['summary']}")
    return state


# ─────────────────────────────────────────────────────────────────
# ERROR HANDLER NODE
# Routes here from any node if state["error"] is set
# ─────────────────────────────────────────────────────────────────
def handle_generation_error(state: AgentState) -> AgentState:
    print(f"[Error Handler] {state.get('error')}")
    state["report"] = {"error": state.get("error", "Unknown error occurred")}
    return state


def check_error(state: AgentState) -> str:
    return "handle_generation_error" if state.get("error") else "continue"


# ─────────────────────────────────────────────────────────────────
# BUILD LANGGRAPH
# START
#   → convert_user_instruction_to_actions  (parse NL + generate code)
#   → execute_test_case                    (run Playwright)
#   → generate_test_report                 (compile report)
#   → END
#
# Error path (from any node):
#   → handle_generation_error → END
# ─────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("convert_user_instruction_to_actions", convert_user_instruction_to_actions)
    graph.add_node("execute_test_case",                   execute_test_case)
    graph.add_node("generate_test_report",                generate_test_report)
    graph.add_node("handle_generation_error",             handle_generation_error)

    graph.set_entry_point("convert_user_instruction_to_actions")

    graph.add_conditional_edges(
        "convert_user_instruction_to_actions", check_error, {
            "continue":                "execute_test_case",
            "handle_generation_error": "handle_generation_error"
        }
    )
    graph.add_conditional_edges(
        "execute_test_case", check_error, {
            "continue":                "generate_test_report",
            "handle_generation_error": "handle_generation_error"
        }
    )

    graph.add_edge("generate_test_report",    END)
    graph.add_edge("handle_generation_error", END)

    return graph.compile()


agent_graph = build_graph()


# ─────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/run-tests", methods=["POST"])
def run_tests():
    data         = request.get_json()
    url          = data.get("url", "").strip()
    instructions = data.get("instructions", "").strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return jsonify({"error": "Please set GEMINI_API_KEY in app.py"}), 400
    if not instructions:
        return jsonify({"error": "Test instructions are required"}), 400

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        initial_state: AgentState = {
            "url":              url,
            "raw_instructions": instructions,
            "enriched_steps":   None,
            "step_details":     None,
            "test_result":      None,
            "report":           None,
            "error":            None,
        }

        final_state = agent_graph.invoke(initial_state)
        report      = final_state.get("report", {})

        if "error" in report and len(report) == 1:
            return jsonify(report), 500

        return jsonify(report)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)