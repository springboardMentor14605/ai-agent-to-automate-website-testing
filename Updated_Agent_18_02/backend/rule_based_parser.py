def normalize_instruction(text):

    return text.lower().strip()

def split_steps(instruction):

    separators = [",", " and "]
    steps = [instruction]
    for sep in separators:
        temp = []
        for step in steps:
            temp.extend(step.split(sep))
        steps = temp

    return [step.strip() for step in steps if step.strip()]

ACTION_KEYWORDS = {
    "open": ["open", "launch", "navigate"],
    "click": ["click", "press", "submit"],
    "fill": ["enter", "type", "fill"],
    "assert": ["verify", "check", "ensure"]
}

def detect_action(step):

    for action, keywords in ACTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in step:
                return action
    return "unknown"

def extract_parameters(action, step):

    if action == "open":
        return {"target": step.replace("open", "").strip()}
    if action == "click":
        return {"target": step.replace("click", "").strip()}
    if action == "fill":
        words = step.split()
        if "as" in words:
            idx = words.index("as")
            # Basic safety check to ensure indices exist
            if idx > 0 and idx + 1 < len(words):
                return {
                    "field": words[idx - 1],
                    "value": words[idx + 1]
                }
    if action == "assert":
        return {"condition": step}
    return {}

def parse_instruction(instruction):

    instruction = normalize_instruction(instruction)
    steps = split_steps(instruction)
    parsed_steps = []
    for step in steps:
        action = detect_action(step)
        params = extract_parameters(action, step)
        parsed_steps.append({
            "action": action,
            "params": params,
            "raw_step": step
        })

    return parsed_steps

if __name__ == "__main__":

    test_case = (
        "Open the login page, "
        "enter username as admin and enter password as 1234, "
        "click login"
    )
    output = parse_instruction(test_case)
    for step in output:
        print(step)
