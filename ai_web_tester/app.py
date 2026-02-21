"""
Flask Web Application
Provides a REST API and serves the HTML frontend for the AI Web Testing Agent.
"""

import os
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

# Lazy import to avoid circular issues
app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory job store (use Redis/DB for production)
jobs: dict = {}
job_counter = 0
job_lock = threading.Lock()


def _run_agent_job(job_id: str, payload: dict):
    """Background thread: run the agent and store the result."""
    from agent import WebTestingAgent
    jobs[job_id]["status"] = "running"

    try:
        agent = WebTestingAgent()
        result = agent.run(
            instruction=payload["instruction"],
            test_name=payload.get("test_name", "AI Web Test"),
            target_url=payload.get("target_url", ""),
            headless=payload.get("headless", True),
            use_llm_debugging=payload.get("use_llm_debugging", False),
            max_retries=payload.get("max_retries", 2)
        )
        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "done"
        jobs[job_id]["final_status"] = result.get("final_status", "UNKNOWN")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Routes ──────────────────────────────────────────────────────────────────── #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a new test job. Returns a job_id for polling."""
    global job_counter
    data = request.get_json(force=True)

    if not data or not data.get("instruction"):
        return jsonify({"error": "Missing 'instruction' field."}), 400

    with job_lock:
        job_counter += 1
        job_id = f"job_{job_counter:04d}"

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "payload": data,
        "result": None,
        "error": None,
        "final_status": None
    }

    thread = threading.Thread(target=_run_agent_job, args=(job_id, data), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    """Poll for job status and results."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    response = {
        "id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "final_status": job.get("final_status"),
        "error": job.get("error")
    }

    if job["status"] == "done" and job.get("result"):
        result = job["result"]
        exec_result = result.get("execution_result") or {}
        response["summary"] = {
            "test_name": exec_result.get("test_name", ""),
            "total_steps": exec_result.get("total_steps", 0),
            "steps_passed": exec_result.get("steps_passed", 0),
            "steps_failed": exec_result.get("steps_failed", 0),
            "execution_time": exec_result.get("execution_time", ""),
            "timestamp": exec_result.get("timestamp", ""),
            "step_results": exec_result.get("step_results", []),
            "debug_suggestion": exec_result.get("debug_suggestion"),
            "report_paths": result.get("report_paths")
        }

    return jsonify(response)


@app.route("/api/jobs")
def api_jobs():
    """List all jobs."""
    return jsonify([
        {"id": j["id"], "status": j["status"], "created_at": j["created_at"],
         "final_status": j.get("final_status")}
        for j in jobs.values()
    ])


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """Serve generated HTML reports."""
    return send_from_directory("reports", filename)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
