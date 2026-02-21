# 📄 llm_assertion_generator.py — Documentation

## Purpose

This module provides the **Gemini LLM client** used by `app.py` to generate Playwright assertions from natural language test expectations. It wraps LangChain's `ChatGoogleGenerativeAI` with a clean interface.

---

## Functions

### `extract_text(content) -> str`

A utility that safely extracts plain text from a Gemini API response, regardless of whether it returns a string or a list of content blocks.

```python
def extract_text(content) -> str
```

**Why it exists:** Gemini can return responses in two formats:
- A plain `str`
- A `list` of dicts like `[{"text": "..."}]`

This function handles both cases so the rest of the code doesn't need to worry about the response format.

**Examples:**
```python
extract_text("Hello world")
# → "Hello world"

extract_text([{"text": "Hello "}, {"text": "world"}])
# → "Hello world"

extract_text(42)
# → "42"
```

---

## Class — `LLMAssertionGenerator`

### Constructor

```python
LLMAssertionGenerator(api_key: str)
```

Initialises the Gemini LLM client using LangChain:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0       # deterministic output — important for code generation
)
```

`temperature=0` is intentional — it makes the LLM output consistent and predictable, which is critical when generating code like Playwright assertions.

---

### `generate_assertions(parsed_step: Dict) -> List[str]`

Generates Playwright assertion strings for a single parsed test step.

**Input:**
```python
{
    "action":   "fill",
    "target":   "#username",
    "expected": "Username field shows student"
}
```

**Output:**
```python
["expect(page.locator('#username')).to_have_value('student')"]
```

**Side effect for `fill` actions:**
If the generated assertion contains `to_have_value(...)`, the method extracts the value and writes it back to `parsed_step["value"]`. This ensures the executor knows what text to type.

---

### `_build_prompt(action, target, expected) -> str` *(private)*

Builds the prompt sent to Gemini for assertion generation.

Key rules enforced in the prompt:
- Output must be a **single line** starting with `expect(`
- Use Python Playwright syntax only (`expect()`, not `assert`)
- For error messages, always use selector `#error`
- No markdown, no explanation — just the assertion line

---

### `_post_process(text: str) -> List[str]` *(private)*

Filters the LLM's raw text response to extract only valid assertion lines.

```python
return [line.strip() for line in text.split("\n") if line.strip().startswith("expect(")]
```

This ensures even if Gemini adds extra commentary, only the actual `expect(...)` lines are returned.

---

## How It's Used in app.py

In `app.py`, `LLMAssertionGenerator` is instantiated in Node 1:

```python
generator = LLMAssertionGenerator(GEMINI_API_KEY)
response = generator.llm.invoke([HumanMessage(content=prompt)])
raw = extract_text(response.content)
```

The full assertion generation is handled inside the single Node 1 LLM call — `generate_assertions()` is available for standalone use but the main pipeline calls the LLM directly via `generator.llm` for efficiency.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-google-genai` | Gemini LLM client |
| `langchain-core` | `HumanMessage` class |
