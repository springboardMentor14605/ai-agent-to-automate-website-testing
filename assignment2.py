import os
import json
import re
import sys
from typing import List, Dict, Union, Any
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END


load_dotenv()
# Use ChatHuggingFace 
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize the base endpoint (do not specify the task here)
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=1000,
    temperature=0.7,
)

# Wrap it in ChatHuggingFace to automatically use the "conversational" task
model = ChatHuggingFace(llm=llm)
SYSTEM_PROMPT = """
You are an instruction parser for an automated web testing system.
Your task:
Convert a natural language test case into structured test commands.
Output ONLY valid JSON.

Do not explain anything.
Do not add extra keys.

Allowed actions:
open
fill
click
assert

Output format:
[
  {
    "action": "<action>",
    "params": {...}
  }
]
"""

# --- Core Logic ---

def clean_json_output(content: str) -> str:
    """
    Helper function to strip Markdown code blocks if the LLM adds them.
    """
    if "```json" in content:
        content = content.replace("```json", "").replace("```", "")
    elif "```" in content:
        content = content.replace("```", "")
    return content.strip()

def llm_parse_instruction(instruction: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sends the system prompt and user instruction to Gemini and handles response formats.
    [span_5](start_span)Refined version of the source function[span_5](end_span).
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction)
    ]

    try:
        
        response = model.invoke(messages)
        content = response.content
        if isinstance(content, dict) and "text" in content:
            cleaned_text = clean_json_output(content["text"])
            return json.loads(cleaned_text)

        if isinstance(content, str):
            cleaned_text = clean_json_output(content)
            return json.loads(cleaned_text)

        if isinstance(content, list):
            return content

        raise TypeError(f"Unexpected Gemini response format: {type(content)}")

    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON", "raw_output": content, "details": str(e)}
    except Exception as e:
        return {"error": "An error occurred during processing", "details": str(e)}

# --- Execution ---

if __name__ == "__main__":
    test_case = """
    Open the signin page.
    Enter username as user1.
    Enter password as 4%23.
    Click the signin button.
    """

    print(f"Processing Test Case:\n{test_case}\n" + "-"*30)

    output = llm_parse_instruction(test_case)

    print(json.dumps(output, indent=2))
