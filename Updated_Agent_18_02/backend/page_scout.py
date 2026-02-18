"""
Page Scout — navigates to a URL in a subprocess and extracts
the login form structure (inputs, buttons, links) so the LLM
can determine the correct selectors for any website.
"""
import os
import sys
import json
import subprocess
import tempfile


def scout_page(url: str, timeout: int = 30000) -> dict:
    """
    Navigate to a URL and extract all form-related elements.

    Returns a dict with:
      - success: bool
      - html_snippet: str  (relevant form HTML)
      - elements: list     (structured input/button data)
      - page_title: str
      - final_url: str
      - error: str | None
    """
    script = _build_scout_script(url, timeout)

    tmp_script = None
    try:
        tmp_script = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False,
            encoding='utf-8'
        )
        tmp_script.write(script)
        tmp_script.close()

        python_exe = sys.executable
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [python_exe, tmp_script.name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout // 1000 + 20,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )

        stdout = proc.stdout.strip()

        if proc.returncode == 0 and stdout:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

        return {
            "success": False,
            "error": proc.stderr.strip() or "Scout failed to produce output",
            "html_snippet": "",
            "elements": [],
            "page_title": "",
            "final_url": url
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Scout timed out after {timeout // 1000 + 20}s",
            "html_snippet": "",
            "elements": [],
            "page_title": "",
            "final_url": url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "html_snippet": "",
            "elements": [],
            "page_title": "",
            "final_url": url
        }
    finally:
        if tmp_script and os.path.exists(tmp_script.name):
            try:
                os.unlink(tmp_script.name)
            except Exception:
                pass


def _build_scout_script(url: str, timeout: int) -> str:
    return f'''# -*- coding: utf-8 -*-
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def scout():
    from playwright.sync_api import sync_playwright

    url = {repr(url)}
    result = {{
        "success": False,
        "html_snippet": "",
        "elements": [],
        "page_title": "",
        "final_url": url,
        "error": None
    }}

    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={{"width": 1280, "height": 720}})
            page = context.new_page()
            page.set_default_timeout({timeout})

            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(2000)

            result["page_title"] = page.title()
            result["final_url"] = page.url

            # ---- Try to find elements in main page first ----
            elements = extract_form_elements(page)

            # ---- If nothing found, check iframes ----
            if not elements:
                for frame in page.frames:
                    if frame == page.main_frame:
                        continue
                    try:
                        frame_elements = extract_form_elements(frame)
                        if frame_elements:
                            elements = frame_elements
                            # Mark that these are inside an iframe
                            for el in elements:
                                el["in_iframe"] = True
                                el["frame_url"] = frame.url
                            break
                    except Exception:
                        continue

            result["elements"] = elements

            # Build a concise HTML snippet for the LLM
            snippet_parts = []
            for el in elements:
                attrs = " ".join(f'{{k}}="{{v}}"' for k, v in el.get("attributes", {{}}).items() if v)
                tag = el.get("tag", "input")
                snippet_parts.append(f"<{{tag}} {{attrs}} />")

            result["html_snippet"] = "\\n".join(snippet_parts)
            result["success"] = len(elements) > 0

    except Exception as e:
        result["error"] = f"{{type(e).__name__}}: {{str(e)}}"
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass

    return result


def extract_form_elements(page_or_frame):
    """Extract all input, button, and submit elements from a page or frame."""
    elements = []

    # Collect input fields (including search)
    inputs = page_or_frame.query_selector_all(
        "input[type='text'], input[type='email'], input[type='password'], "
        "input[type='tel'], input[type='number'], input[type='submit'], "
        "input[type='search'], input:not([type]), textarea, "
        "[role='searchbox'], [role='textbox'], [contenteditable='true']"
    )
    for el in inputs:
        try:
            if not el.is_visible():
                continue
            attrs = {{}}
            for attr in ["id", "name", "type", "placeholder", "class", "aria-label", "autocomplete", "data-testid", "role", "title"]:
                val = el.get_attribute(attr)
                if val:
                    attrs[attr] = val
            input_type = attrs.get("type", "text")
            elements.append({{
                "tag": "input",
                "role": classify_input(attrs),
                "attributes": attrs,
                "selector": build_selector(el, attrs),
                "input_type": input_type
            }})
        except Exception:
            continue

    # Collect buttons (expanded selectors)
    buttons = page_or_frame.query_selector_all(
        "button, input[type='submit'], [role='button'], a.btn, a.login-btn, "
        "a[href*='login'], a[href*='signin'], a[href*='cart'], "
        "[type='submit'], button[type='submit'], "
        "a[href*='checkout'], [data-action]"
    )
    for el in buttons:
        try:
            if not el.is_visible():
                continue
            attrs = {{}}
            for attr in ["id", "name", "type", "class", "aria-label", "data-testid", "value", "title", "href"]:
                val = el.get_attribute(attr)
                if val:
                    attrs[attr] = val
            text = el.inner_text().strip()[:50]
            tag_name = el.evaluate("el => el.tagName.toLowerCase()")
            elements.append({{
                "tag": tag_name,
                "role": "submit",
                "attributes": attrs,
                "text": text,
                "selector": build_selector(el, attrs, is_button=True),
                "input_type": "button"
            }})
        except Exception:
            continue

    # Collect select dropdowns
    selects = page_or_frame.query_selector_all("select")
    for el in selects:
        try:
            if not el.is_visible():
                continue
            attrs = {{}}
            for attr in ["id", "name", "class", "aria-label", "data-testid"]:
                val = el.get_attribute(attr)
                if val:
                    attrs[attr] = val
            elements.append({{
                "tag": "select",
                "role": "dropdown",
                "attributes": attrs,
                "selector": build_selector(el, attrs),
                "input_type": "select"
            }})
        except Exception:
            continue

    # Collect important links (navigation)
    links = page_or_frame.query_selector_all(
        "a[href*='cart'], a[href*='account'], a[href*='profile'], "
        "a[href*='wishlist'], a[href*='order']"
    )
    for el in links:
        try:
            if not el.is_visible():
                continue
            attrs = {{}}
            for attr in ["id", "href", "class", "aria-label", "data-testid", "title"]:
                val = el.get_attribute(attr)
                if val:
                    attrs[attr] = val
            text = el.inner_text().strip()[:50]
            elements.append({{
                "tag": "a",
                "role": "link",
                "attributes": attrs,
                "text": text,
                "selector": build_selector(el, attrs),
                "input_type": "link"
            }})
        except Exception:
            continue

    return elements


def classify_input(attrs):
    """Guess if an input is for username, password, email, phone, or search."""
    input_type = (attrs.get("type") or "").lower()
    name = (attrs.get("name") or "").lower()
    placeholder = (attrs.get("placeholder") or "").lower()
    autocomplete = (attrs.get("autocomplete") or "").lower()
    aria_label = (attrs.get("aria-label") or "").lower()
    role = (attrs.get("role") or "").lower()
    title = (attrs.get("title") or "").lower()
    all_text = f"{{name}} {{placeholder}} {{autocomplete}} {{aria_label}} {{role}} {{title}}"

    if input_type == "password":
        return "password"
    if input_type == "search" or role == "searchbox" or any(kw in all_text for kw in ["search", "find", "query", "lookup"]):
        return "search"
    if any(kw in all_text for kw in ["email", "e-mail", "mail"]):
        return "email"
    if any(kw in all_text for kw in ["phone", "mobile", "tel", "cell"]):
        return "phone"
    if any(kw in all_text for kw in ["user", "login", "account", "id", "name"]):
        return "username"
    if input_type == "text" or input_type == "":
        return "text"    # generic text field
    return "unknown"


def build_selector(el, attrs, is_button=False):
    """Build the best CSS selector for an element."""
    # Prefer ID
    if attrs.get("id"):
        return f"#{{attrs['id']}}"
    # Then data-testid
    if attrs.get("data-testid"):
        return f"[data-testid='{{attrs['data-testid']}}']"
    # Then name
    if attrs.get("name"):
        tag = "input" if not is_button else "*"
        return f"{{tag}}[name='{{attrs['name']}}']"
    # Then type + placeholder
    if attrs.get("placeholder"):
        return f"input[placeholder='{{attrs['placeholder']}}']"
    # Fallback: try aria-label
    if attrs.get("aria-label"):
        return f"[aria-label='{{attrs['aria-label']}}']"
    # Fallback: for buttons/submit inputs, use value attribute
    if attrs.get("value") and is_button:
        return f"input[value='{{attrs['value']}}']"
    # Fallback: for submit inputs, use type
    if attrs.get("type") == "submit":
        return "input[type='submit']"
    # Fallback: try class (only for buttons)
    if is_button and attrs.get("class"):
        cls = attrs['class'].split()[0]
        return f"button.{{cls}}, input.{{cls}}"
    return ""


if __name__ == "__main__":
    res = scout()
    print(json.dumps(res, ensure_ascii=True))
'''
