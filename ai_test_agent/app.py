from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run-test', methods=['POST'])
def run_test():
    data = request.json
    instruction = data.get('instruction')
    
    from core.parser import InstructionParser
    from core.generator import CodeGenerator
    from core.executor import Executor

    # 1. Parse instruction
    parser = InstructionParser()
    actions = parser.parse(instruction)
    
    if not actions:
        return jsonify({
            "status": "error",
            "message": "Could not parse instruction or no actions generated.",
            "logs": [">> Parser failed to interpret instruction."],
            "report": "FAILED"
        })

    # 2. Generate Code
    generator = CodeGenerator()
    script = generator.generate(actions)
    
    # 3. Execute Script
    executor = Executor()
    result = executor.execute(script)
    
    status = "SUCCESS" if result["success"] else "FAILURE"
    
    response = {
        "status": status, 
        "message": f"Execution finished with status: {status}",
        "logs": [f">> AI_INTENT: Parsed {len(actions)} actions: " + ", ".join([a['action'] for a in actions])] + result["logs"] + result["errors"],
        "report": f"Test Status: {status}\n\nActions Performed: {len(actions)}\n\nDetails:\n" + "\n".join(result["logs"])
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
