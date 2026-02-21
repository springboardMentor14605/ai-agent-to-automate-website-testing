# 📄 app.py — Documentation

## Purpose

`app.py` is the **main entry point** of AutoTest AI. It does three things:
1. Defines the **LangGraph agent pipeline** (3 nodes + error handler)
2. Hosts the **Flask web server** with API routes
3. Wires everything together so a single API call triggers the full test run

---

## Configuration

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

Set your Google Gemini API key here before running. Get one free at [aistudio.google.com](https://aistudio.google.com).

---

## AgentState

```python
class AgentState(TypedDict):
    url:              str
    raw_instructions: str
    enriched_steps:   Optional[List]
    step_details:     Optional[List]
    test_result:      Optional[dict]
    report:           Optional[dict]
    error:            Optional[str]
```

This is the **shared state** passed between every node in the LangGraph pipeline. Each node reads from it and writes back to it.

---

## Node 1 — `convert_user_instruction_to_actions`

### What it does
Sends a single prompt to Google Gemini that asks it to:
- Parse the user's natural language instructions into structured steps
- Generate a Playwright Python assertion for each step

### Input (from state)
- `state["url"]` — the target website
- `state["raw_instructions"]` — the user's plain English test description

### Output (written to state)
- `state["enriched_steps"]` — list of steps used by Playwright executor
- `state["step_details"]` — UI-friendly copy used for reporting

### Step format

Each step in `enriched_steps` looks like:
```python
{
    "action":     "fill",           # open | fill | click | wait
    "target":     "#username",      # CSS selector or URL
    "value":      "student",        # text to type (fill only)
    "expected":   "Username field shows student",
    "assertions": ["expect(page.locator('#username')).to_have_value('student')"]
}
```

### Guarantees
- Always ensures the **first step is `action="open"`** with the correct URL
- Strips markdown code fences from LLM response before JSON parsing
- On JSON parse failure, sets `state["error"]` to route to error handler

---

## Node 2 — `execute_test_case`

### What it does
Passes `enriched_steps` to `PlaywrightExecutor` which runs them in a real headless Chromium browser.

### Input (from state)
- `state["enriched_steps"]`
- `state["step_details"]`

### Output (written to state)
- `state["test_result"]` — dict containing status, steps_executed, per-step PASS/FAIL, error messages

### Behaviour on failure
If a step fails, the executor **continues running remaining steps** (does not abort). The overall status becomes `"FAIL"` but all steps are attempted.

---

## Node 3 — `generate_test_report`

### What it does
Compiles the final report dictionary that gets sent back to the frontend.

### Output structure
```python
{
    "url":            "https://example.com",
    "instructions":   "...",
    "total_steps":    4,
    "steps_executed": 4,
    "status":         "PASS",   # or "FAIL"
    "error":          None,
    "steps":          [...],    # per-step details with status
    "summary": {
        "passed":  3,
        "failed":  1,
        "skipped": 0
    }
}
```

---

## Error Handler — `handle_generation_error`

If `state["error"]` is set by any node, the conditional edge routes here instead of the next node. It returns a minimal error report:

```python
{"error": "description of what went wrong"}
```

---

## LangGraph Graph Structure

```python
graph.set_entry_point("convert_user_instruction_to_actions")

# After Node 1 — check for error
convert → (no error) → execute_test_case
        → (error)    → handle_generation_error → END

# After Node 2 — check for error
execute → (no error) → generate_test_report → END
        → (error)    → handle_generation_error → END
```

---

## Flask Routes

### `GET /`
Serves `static/index.html` — the frontend UI.

### `POST /api/run-tests`

**Request body:**
```json
{
  "url": "https://example.com",
  "instructions": "Enter username as student\nClick submit"
}
```

**Validation:**
- URL is required
- Instructions are required
- `GEMINI_API_KEY` must be set (not the placeholder)
- Automatically prepends `https://` if missing from URL

**Response:** Full report JSON (see Node 3 output above)

**Error response (500):**
```json
{ "error": "description" }
```
