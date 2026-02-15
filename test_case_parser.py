import os
import json
import re
import sys
from typing import List, Dict, Union, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


os.environ["GOOGLE_API_KEY"] = ""
# Use ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

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
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content
        
        # Handle different response formats
        if isinstance(content, list):
            # If content is a list with dict items containing 'text' field
            if len(content) > 0 and isinstance(content[0], dict) and 'text' in content[0]:
                text_content = content[0]['text']
                cleaned_text = clean_json_output(text_content)
                return json.loads(cleaned_text)
            return content
        
        if isinstance(content, str):
            cleaned_text = clean_json_output(content)
            return json.loads(cleaned_text)
        
        raise TypeError(f"Unexpected Gemini response format: {type(content)}")
        
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON", "raw_output": str(content), "details": str(e)}
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