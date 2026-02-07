import os
import subprocess
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set as an environment variable")

# LANGGRAPH STATE
class AgentState(TypedDict):
    instructions: str
    playwright_code: str
    file_path: str

# GEMINI MODEL
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

# NODE 1: LLM GENERATES PLAYWRIGHT CODE
def playwright_code_generator(state: AgentState) -> AgentState:
    system_prompt = """
You are an expert automation engineer.

Generate a COMPLETE Playwright Python script using playwright.sync_api.

Rules:
- Output ONLY valid Python code
- Do NOT explain anything
- Use sync_playwright
- Close the browser at the end
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["instructions"])
    ])

    content = response.content

    # Extract text properly (Gemini may return list)
    if isinstance(content, str):
        code = content
    elif isinstance(content, list):
        code = content[0]["text"]
    else:
        raise ValueError("Unexpected Gemini response format")

    # Remove markdown formatting if present
    code = code.replace("```python", "").replace("```", "").strip()

    state["playwright_code"] = code
    return state

# NODE 2: SAVE CODE TO FILE
def save_code_node(state: AgentState) -> AgentState:
    file_path = "generated_playwright_test.py"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(state["playwright_code"])

    state["file_path"] = file_path

    print("\n--- GENERATED PLAYWRIGHT CODE ---\n")
    print(state["playwright_code"])
    print("\n--- CODE SAVED AS:", file_path, "---\n")

    return state

# NODE 3: EXECUTE SAVED FILE
def execute_code_node(state: AgentState) -> AgentState:
    print("Executing Playwright script...\n")
    subprocess.run([sys.executable, state["file_path"]], check=True)
    return state

# LANGGRAPH WORKFLOW
graph = StateGraph(AgentState)

graph.add_node("generator", playwright_code_generator)
graph.add_node("save", save_code_node)
graph.add_node("execute", execute_code_node)

graph.set_entry_point("generator")
graph.add_edge("generator", "save")
graph.add_edge("save", "execute")
graph.add_edge("execute", END)

app = graph.compile()

# RUN
if __name__ == "__main__":
    instructions = """
    Write a Playwright test in Python to test this website https://practicetestautomation.com/practice-test-login/
    """

    app.invoke({
        "instructions": instructions,
        "playwright_code": "",
        "file_path": ""
    })