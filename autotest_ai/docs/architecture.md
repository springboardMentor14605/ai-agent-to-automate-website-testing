# 🏗️ AutoTest AI — System Architecture

## Overview

AutoTest AI is a 3-node agentic pipeline built with **LangGraph**. The user provides a URL and plain English test instructions; the system converts them into structured steps, runs them in a real browser, and returns a detailed pass/fail report.

---

## High-Level Flow

```
User (Browser UI)
      │
      │  POST /api/run-tests
      ▼
┌─────────────────┐
│   Flask Server  │  (app.py)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Node 1: convert_user_instruction_to_actions  │   │
│  │  → 1 Gemini LLM call                        │   │
│  │  → Outputs: structured steps + assertions   │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                           │
│  ┌──────────────────────▼───────────────────────┐   │
│  │ Node 2: execute_test_case                    │   │
│  │  → Playwright headless Chromium              │   │
│  │  → Runs each step, marks PASS/FAIL           │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                           │
│  ┌──────────────────────▼───────────────────────┐   │
│  │ Node 3: generate_test_report                 │   │
│  │  → Compiles final report dict                │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                           │
│            ┌────────────▼────────────┐              │
│            │  handle_generation_error│ (if error)   │
│            └─────────────────────────┘              │
└─────────────────────────────────────────────────────┘
         │
         ▼
   JSON Report → Flask → Browser UI
```

---

## LangGraph State

All nodes share a single typed state object (`AgentState`):

| Field | Type | Description |
|---|---|---|
| `url` | `str` | Target website URL |
| `raw_instructions` | `str` | User's plain English input |
| `enriched_steps` | `List` | Parsed steps with assertions (for Playwright) |
| `step_details` | `List` | UI-friendly copy of steps for reporting |
| `test_result` | `dict` | Raw output from Playwright executor |
| `report` | `dict` | Final compiled report sent to frontend |
| `error` | `str` | Error message if any node fails |

---

## Error Handling

Every node transition has a conditional edge:

```
Node → check_error() → "continue" → next node
                     → "handle_generation_error" → END
```

If `state["error"]` is set at any point, the pipeline short-circuits to the error handler which returns a minimal error report to the UI.

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS |
| Backend | Python, Flask |
| Agent Framework | LangGraph |
| LLM | Google Gemini via LangChain |
| Browser Automation | Playwright (headless Chromium) |

---

## File Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, LangGraph graph definition, all 3 nodes |
| `llm_assertion_generator.py` | Gemini LLM client wrapper |
| `playwright_executor.py` | Browser step execution and assertion evaluation |
| `static/index.html` | Frontend UI |
