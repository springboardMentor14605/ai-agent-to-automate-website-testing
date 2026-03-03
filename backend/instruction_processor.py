"""
Instruction Processor — Parses natural language commands into
structured Playwright test steps using LLM.

Handles intent detection (add to cart, remove, navigate, fill, click, etc.),
entity extraction (product names, quantities, selectors), and maps
commands to executable Playwright actions.
"""
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

api_key = os.getenv("ABHAY_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY (or ABHAY_API_KEY) not found in environment")

os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ==================================================
# SYSTEM PROMPT FOR INSTRUCTION PARSING
# ==================================================

INSTRUCTION_SYSTEM_PROMPT = """
You are an intelligent instruction parser for an automated web testing and interaction system.
Your job is to convert natural language commands into structured Playwright test steps.

You will receive:
1. A natural language instruction from the user (e.g., "Add 2 iPhones to cart")
2. The target website URL
3. Page structure data (form elements, buttons, links discovered by scouting the page)

Output ONLY valid JSON. No markdown. No comments. No explanation.

Allowed actions:
- open: Navigate to a URL. { "action": "open", "target": "<url>", "value": "" }
- click: Click an element. { "action": "click", "target": "<css-selector>", "value": "" }
- fill: Type into an input. { "action": "fill", "target": "<css-selector>", "value": "<text>" }
- wait: Wait for a duration. { "action": "wait", "target": "", "value": "<milliseconds>" }
- select: Select from dropdown. { "action": "select", "target": "<css-selector>", "value": "<option-value>" }
- scroll: Scroll the page. { "action": "scroll", "target": "<css-selector or empty>", "value": "<up|down>" }
- press: Press a keyboard key. { "action": "press", "target": "", "value": "<key-name like Enter, Tab, Escape>" }

IMPORTANT RULES:
1. Use EXACT selectors from the page structure if provided.
2. Do NOT invent selectors — only use selectors from the page elements list.
3. If a selector is not available, use descriptive text-based selectors like:
   - text="Button Text" (for clicking elements by their visible text)
   - [aria-label="description"] (for accessible elements)
4. For "add to cart" type actions, look for buttons containing text like "Add to Cart", "Buy", "Add".
5. If a quantity needs to be set, look for quantity input fields and fill them.
6. For product searches, look for search inputs and fill them, then submit.
7. Always include a wait step (500-1000ms) after navigation or important clicks.
8. Think step-by-step: navigate → find product → set quantity → add to cart.

ALSO include an "intent" field in your response describing the detected intent.
Valid intents: navigate, search, add_to_cart, remove_from_cart, update_quantity,
clear_cart, fill_form, click_element, login, checkout, custom

Output format:
{
  "intent": "<detected_intent>",
  "description": "<human-readable description of what will be done>",
  "entities": {
    "product": "<product name if applicable>",
    "quantity": <quantity if applicable>,
    "action_type": "<specific action type>"
  },
  "steps": [
    {
      "action": "<action>",
      "target": "<target>",
      "value": "<value>",
      "description": "<human-readable step description>"
    }
  ]
}
"""


def clean_json_output(text: str) -> str:
    """Remove accidental markdown/code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
    return text.strip()


def parse_instruction(instruction: str, url: str, scout_data: dict = None) -> dict:
    """
    Parse a natural language instruction into structured test steps.

    Args:
        instruction: Natural language command (e.g., "Add 2 iPhones to cart")
        url: Target website URL
        scout_data: Optional page structure data from scouting

    Returns:
        Dict with intent, description, entities, and steps
    """
    # Build context message with page structure
    context_parts = [
        f"Target URL: {url}",
        f"User instruction: {instruction}",
    ]

    if scout_data and scout_data.get("success"):
        elements = scout_data.get("elements", [])
        if elements:
            element_descriptions = []
            for el in elements:
                role = el.get("role", "unknown")
                selector = el.get("selector", "")
                text = el.get("text", "")
                attrs = el.get("attributes", {})
                desc = f"  - Role: {role}, Selector: {selector}"
                if text:
                    desc += f", Text: {text}"
                if attrs.get("placeholder"):
                    desc += f", Placeholder: {attrs['placeholder']}"
                element_descriptions.append(desc)

            context_parts.append(
                "\n--- PAGE STRUCTURE (discovered by scouting) ---\n"
                f"Page title: {scout_data.get('page_title', '')}\n"
                f"Elements:\n" + "\n".join(element_descriptions)
            )
    else:
        context_parts.append(
            "\nNote: No page structure data available. Use general web selectors "
            "and text-based selectors where possible."
        )

    user_message = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=INSTRUCTION_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content

        # Handle various response formats
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
        except json.JSONDecodeError:
            # Try to extract JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                parsed = json.loads(cleaned[start:end + 1])
            else:
                raise ValueError(f"Could not parse LLM response as JSON")

        # Validate structure
        if "steps" not in parsed:
            raise ValueError("LLM response missing 'steps' field")

        return {
            "success": True,
            "intent": parsed.get("intent", "custom"),
            "description": parsed.get("description", "Executing instruction"),
            "entities": parsed.get("entities", {}),
            "steps": parsed.get("steps", []),
        }

    except Exception as e:
        return {
            "success": False,
            "intent": "error",
            "description": f"Failed to parse instruction: {str(e)}",
            "entities": {},
            "steps": [],
            "error": str(e),
        }


def validate_instruction(instruction: str) -> dict:
    """
    Quick validation of user instruction before processing.

    Returns:
        Dict with is_valid flag and optional error message.
    """
    if not instruction or not instruction.strip():
        return {
            "is_valid": False,
            "error": "Instruction cannot be empty."
        }

    if len(instruction.strip()) < 5:
        return {
            "is_valid": False,
            "error": "Instruction is too short. Please provide more detail."
        }

    if len(instruction) > 2000:
        return {
            "is_valid": False,
            "error": "Instruction is too long. Please keep it under 2000 characters."
        }

    return {"is_valid": True}


# Example command suggestions for the frontend
EXAMPLE_COMMANDS = [
    {
        "category": "Navigation",
        "examples": [
            "Go to the products page",
            "Navigate to checkout",
            "Open the settings page",
        ]
    },
    {
        "category": "Cart Actions",
        "examples": [
            "Add 2 iPhones to cart",
            "Remove AirPods from cart",
            "Increase MacBook quantity to 3",
            "Clear my cart",
        ]
    },
    {
        "category": "Form Interactions",
        "examples": [
            "Fill in the search box with 'laptop'",
            "Select 'Large' from the size dropdown",
            "Enter my email as test@example.com",
        ]
    },
    {
        "category": "Click Actions",
        "examples": [
            "Click the 'Buy Now' button",
            "Click on the first product",
            "Click the menu icon",
        ]
    },
]
