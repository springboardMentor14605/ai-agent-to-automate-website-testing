"""
Assertion & Reporting Module
Compiles test execution results into structured, human-readable reports.
Outputs to console, JSON file, and HTML file.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Test Report – {test_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

  :root {{
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent-green: #00ff88;
    --accent-red: #ff4060;
    --accent-blue: #4d9fff;
    --accent-yellow: #ffd166;
    --text: #e0e0f0;
    --muted: #6b6b8a;
    --pass: #00ff88;
    --fail: #ff4060;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 2rem;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 2rem;
  }}

  .title-block h1 {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #fff;
  }}

  .title-block .subtitle {{
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.8rem;
    margin-top: 0.4rem;
  }}

  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.1em;
  }}

  .badge.pass {{ background: rgba(0,255,136,0.12); color: var(--pass); border: 1px solid rgba(0,255,136,0.3); }}
  .badge.fail {{ background: rgba(255,64,96,0.12); color: var(--fail); border: 1px solid rgba(255,64,96,0.3); }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }}

  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
  }}

  .stat-card .label {{
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }}

  .stat-card .value {{
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
  }}

  .stat-card.green .value {{ color: var(--accent-green); }}
  .stat-card.red .value {{ color: var(--accent-red); }}
  .stat-card.blue .value {{ color: var(--accent-blue); }}
  .stat-card.yellow .value {{ color: var(--accent-yellow); }}

  .section-title {{
    font-family: var(--mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 1rem;
  }}

  .step-list {{ display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 2.5rem; }}

  .step {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}

  .step-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1.2rem;
    cursor: pointer;
    user-select: none;
  }}

  .step-header:hover {{ background: rgba(255,255,255,0.03); }}

  .step-num {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    min-width: 2rem;
  }}

  .step-status {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  .step-status.pass {{ background: var(--pass); box-shadow: 0 0 6px var(--pass); }}
  .step-status.fail {{ background: var(--fail); box-shadow: 0 0 6px var(--fail); }}

  .step-action {{
    font-family: var(--mono);
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(77,159,255,0.15);
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex-shrink: 0;
  }}

  .step-desc {{ flex: 1; font-size: 0.9rem; color: var(--text); }}

  .step-time {{
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    flex-shrink: 0;
  }}

  .step-detail {{
    display: none;
    padding: 0.8rem 1.2rem 1rem 4.2rem;
    border-top: 1px solid var(--border);
    background: rgba(0,0,0,0.2);
  }}

  .step.open .step-detail {{ display: block; }}

  .detail-row {{
    display: flex;
    gap: 0.8rem;
    margin-bottom: 0.3rem;
    font-family: var(--mono);
    font-size: 0.78rem;
  }}

  .detail-row .key {{ color: var(--muted); min-width: 6rem; flex-shrink: 0; }}
  .detail-row .val {{ color: var(--text); word-break: break-all; }}

  .assertions {{ margin-top: 0.8rem; }}
  .assertions-title {{ font-family: var(--mono); font-size: 0.7rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.4rem; }}

  .assertion-line {{
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.3rem 0;
    font-family: var(--mono);
    font-size: 0.76rem;
  }}

  .assertion-line .icon {{ flex-shrink: 0; margin-top: 1px; }}
  .assertion-line.pass .icon {{ color: var(--pass); }}
  .assertion-line.fail .icon {{ color: var(--fail); }}
  .assertion-line .code {{ color: var(--accent-blue); word-break: break-all; }}
  .assertion-line .err {{ color: var(--fail); font-size: 0.7rem; margin-top: 0.2rem; }}

  .error-block {{
    background: rgba(255,64,96,0.08);
    border: 1px solid rgba(255,64,96,0.25);
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 2.5rem;
  }}

  .error-block .err-title {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--fail);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
  }}

  .error-block pre {{
    font-family: var(--mono);
    font-size: 0.8rem;
    color: #ff8090;
    white-space: pre-wrap;
    word-break: break-all;
  }}

  .debug-block {{
    background: rgba(255,209,102,0.06);
    border: 1px solid rgba(255,209,102,0.2);
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 2.5rem;
  }}

  .debug-block .debug-title {{
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent-yellow);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
  }}

  .debug-block p {{ font-size: 0.9rem; line-height: 1.7; white-space: pre-wrap; }}

  footer {{
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>{test_name}</h1>
    <div class="subtitle">Generated {timestamp} &nbsp;·&nbsp; Execution time: {execution_time}</div>
  </div>
  <div class="badge {status_class}">{status_icon} {status}</div>
</header>

<div class="stats-grid">
  <div class="stat-card blue"><div class="label">Total Steps</div><div class="value">{total_steps}</div></div>
  <div class="stat-card green"><div class="label">Passed</div><div class="value">{steps_passed}</div></div>
  <div class="stat-card red"><div class="label">Failed</div><div class="value">{steps_failed}</div></div>
  <div class="stat-card yellow"><div class="label">Pass Rate</div><div class="value">{pass_rate}%</div></div>
</div>

{error_section}
{debug_section}

<div class="section-title">Step Results</div>
<div class="step-list">
{steps_html}
</div>

<footer>
  <span>AI Web Testing Agent</span>
  <span>Report generated {timestamp}</span>
</footer>

<script>
document.querySelectorAll('.step-header').forEach(h => {{
  h.addEventListener('click', () => h.closest('.step').classList.toggle('open'));
}});
</script>
</body>
</html>
"""


class ReportGenerator:
    """Generates test reports in console, JSON, and HTML formats."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def generate(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all report formats for a test result.

        Returns:
            dict with keys: console, json_path, html_path
        """
        console_text = self._console_report(result)
        json_path = self._save_json(result)
        html_path = self._save_html(result)

        print(console_text)
        return {
            "console": console_text,
            "json_path": json_path,
            "html_path": html_path
        }

    # ------------------------------------------------------------------ #
    # Console
    # ------------------------------------------------------------------ #

    def _console_report(self, r: Dict) -> str:
        lines = []
        w = 70

        lines.append("\n" + "=" * w)
        lines.append(f"  TEST REPORT : {r['test_name']}")
        lines.append(f"  Timestamp   : {r['timestamp']}")
        lines.append("=" * w)

        status_icon = "✅" if r["status"] == "PASS" else "❌"
        lines.append(f"\n  {status_icon}  Status     : {r['status']}")
        lines.append(f"  📊  Steps      : {r['steps_executed']}/{r['total_steps']}")
        lines.append(f"  ✔   Passed     : {r['steps_passed']}")
        lines.append(f"  ✖   Failed     : {r['steps_failed']}")

        total = r["total_steps"]
        passed = r["steps_passed"]
        rate = f"{(passed / total * 100):.1f}" if total else "0.0"
        lines.append(f"  📈  Pass Rate  : {rate}%")
        lines.append(f"  ⏱   Duration  : {r['execution_time']}")

        lines.append("\n" + "-" * w)
        lines.append("  STEP DETAILS")
        lines.append("-" * w)

        for step in r.get("step_results", []):
            icon = "✅" if step["status"] == "PASS" else "❌"
            action = f"[{step['action'].upper()}]"
            lines.append(f"\n  {icon} Step {step['step_number']:>2} {action:<10} {step['description']}")

            if step.get("target"):
                lines.append(f"           Target : {step['target']}")
            if step.get("value"):
                lines.append(f"           Value  : {step['value']}")

            for detail in step.get("assertion_details", []):
                icon2 = "  ✓" if detail["status"] == "PASS" else "  ✗"
                code = detail["assertion"][:60] + ("…" if len(detail["assertion"]) > 60 else "")
                lines.append(f"           {icon2} {code}")
                if detail.get("error"):
                    lines.append(f"               ↳ {detail['error']}")

            if step.get("error"):
                lines.append(f"           ⚠ Error: {step['error']}")

        if r.get("error"):
            lines.append(f"\n{'='*w}")
            lines.append(f"  FAILURE AT STEP {r['failed_step']}: {r['error']}")

        if r.get("debug_suggestion"):
            lines.append(f"\n{'─'*w}")
            lines.append("  🤖 AI DEBUG SUGGESTIONS")
            lines.append("─" * w)
            for line in r["debug_suggestion"].split("\n"):
                if line.strip():
                    lines.append(f"  {line}")

        lines.append("\n" + "=" * w + "\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #

    def _save_json(self, result: Dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = result["test_name"].replace(" ", "_").lower()
        path = os.path.join(self.output_dir, f"{name}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------ #
    # HTML
    # ------------------------------------------------------------------ #

    def _save_html(self, result: Dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = result["test_name"].replace(" ", "_").lower()
        path = os.path.join(self.output_dir, f"{name}_{ts}.html")

        total = result["total_steps"] or 1
        rate = f"{(result['steps_passed'] / total * 100):.1f}"
        status = result["status"]
        status_class = "pass" if status == "PASS" else "fail"
        status_icon = "✓" if status == "PASS" else "✗"

        steps_html = self._render_steps(result.get("step_results", []))
        error_section = self._render_error(result)
        debug_section = self._render_debug(result)

        html = HTML_TEMPLATE.format(
            test_name=result["test_name"],
            timestamp=result.get("timestamp", ""),
            execution_time=result.get("execution_time", ""),
            status=status,
            status_class=status_class,
            status_icon=status_icon,
            total_steps=result["total_steps"],
            steps_passed=result["steps_passed"],
            steps_failed=result["steps_failed"],
            pass_rate=rate,
            steps_html=steps_html,
            error_section=error_section,
            debug_section=debug_section
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def _render_steps(self, steps: List[Dict]) -> str:
        parts = []
        for step in steps:
            status = step["status"]
            css = "pass" if status == "PASS" else "fail"
            action_html = f'<span class="step-action">{step["action"]}</span>'

            assertions_html = ""
            if step.get("assertion_details"):
                rows = []
                for d in step["assertion_details"]:
                    ac = "pass" if d["status"] == "PASS" else "fail"
                    icon = "✓" if d["status"] == "PASS" else "✗"
                    err_html = f'<div class="err">{d["error"]}</div>' if d.get("error") else ""
                    code = d["assertion"].replace("<", "&lt;").replace(">", "&gt;")
                    rows.append(
                        f'<div class="assertion-line {ac}">'
                        f'<span class="icon">{icon}</span>'
                        f'<div><span class="code">{code}</span>{err_html}</div>'
                        f'</div>'
                    )
                assertions_html = (
                    '<div class="assertions">'
                    '<div class="assertions-title">Assertions</div>'
                    + "".join(rows) +
                    '</div>'
                )

            target_html = f'<div class="detail-row"><span class="key">Target</span><span class="val">{step.get("target","")}</span></div>' if step.get("target") else ""
            value_html  = f'<div class="detail-row"><span class="key">Value</span><span class="val">{step.get("value","")}</span></div>'  if step.get("value")  else ""
            error_html  = f'<div class="detail-row"><span class="key" style="color:#ff4060">Error</span><span class="val" style="color:#ff8090">{step.get("error","")}</span></div>' if step.get("error") else ""

            parts.append(
                f'<div class="step">'
                f'<div class="step-header">'
                f'<span class="step-num">#{step["step_number"]:02d}</span>'
                f'<span class="step-status {css}"></span>'
                f'{action_html}'
                f'<span class="step-desc">{step["description"]}</span>'
                f'</div>'
                f'<div class="step-detail">'
                f'{target_html}{value_html}{error_html}'
                f'{assertions_html}'
                f'</div>'
                f'</div>'
            )
        return "\n".join(parts)

    def _render_error(self, result: Dict) -> str:
        if not result.get("error"):
            return ""
        err = result["error"].replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<div class="error-block">'
            f'<div class="err-title">✗ Test Failed at Step {result.get("failed_step","?")}</div>'
            f'<pre>{err}</pre>'
            f'</div>'
        )

    def _render_debug(self, result: Dict) -> str:
        if not result.get("debug_suggestion"):
            return ""
        suggestion = result["debug_suggestion"].replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<div class="debug-block">'
            f'<div class="debug-title">🤖 AI Debug Suggestions</div>'
            f'<p>{suggestion}</p>'
            f'</div>'
        )
