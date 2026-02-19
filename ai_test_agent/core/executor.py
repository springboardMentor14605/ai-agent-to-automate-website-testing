import subprocess
import sys
import os

import tempfile

class Executor:
    def execute(self, script_content):
        """
        Executes the generated Python script in a separate process.
        """
        # Save script to a temporary file in the system temp directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            script_path = tf.name
            tf.write(script_content)
            
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=60 # Timeout to prevent indefinite hanging
            )
            
            logs = result.stdout.splitlines()
            errors = result.stderr.splitlines()
            
            return {
                "success": result.returncode == 0,
                "logs": logs,
                "errors": errors
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "logs": [],
                "errors": ["Execution timed out."]
            }
        except Exception as e:
            return {
                "success": False,
                "logs": [],
                "errors": [str(e)]
            }
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
