class CodeGenerator:
    def generate(self, actions):
        """
        Generates a complete Python Playwright script from a list of actions.
        """
        script_lines = [
            "from playwright.sync_api import sync_playwright",
            "import sys",
            "",
            "def run():",
            "    with sync_playwright() as p:",
            "        browser = p.chromium.launch(headless=True)", # Headless as requested
            "        page = browser.new_page()",
            "        try:",
        ]
        
        for action in actions:
            act = action.get("action")
            if act == "goto":
                script_lines.append(f"            print(f'>> Navigating to {action['url']}')")
                script_lines.append(f"            page.goto('{action['url']}')")
            elif act == "fill":
                script_lines.append(f'            print(f">> Filling {action["selector"]} with {action["value"]}")')
                script_lines.append(f'            page.fill("{action["selector"]}", "{action["value"]}")')
            elif act == "click":
                script_lines.append(f'            print(f">> Clicking {action["selector"]}")')
                script_lines.append(f'            page.click("{action["selector"]}")')
            elif act == "wait_for_selector":
                script_lines.append(f'            print(f">> Waiting for {action["selector"]}")')
                script_lines.append(f'            page.wait_for_selector("{action["selector"]}")')
            elif act == "check_text":
                script_lines.append(f'            print(f">> Verifying text in {action["selector"]}")')
                # Use inner_text() to check visible text only (ignoring script/style content)
                script_lines.append(f'            content = page.inner_text("{action["selector"]}")')
                script_lines.append(f'            assert "{action["expected"]}" in content, f"Expected \'{action["expected"]}\' but found \'{{content}}\'"')
                script_lines.append(f'            print(">> ASSERTION PASSED: {action["expected"]} found.")')
            elif act == "check_title":
                script_lines.append(f"            print(f'>> Checking title is {action['expected']}')")
                script_lines.append(f"            title = page.title()")
                script_lines.append(f"            assert '{action['expected']}'.lower() in title.lower(), f'Expected \"{action['expected']}\" in title but found \"{{title}}\"'")
                script_lines.append(f"            print('>> ASSERTION PASSED: Title contains {action['expected']}')")
            
            elif act == "press":
                script_lines.append(f"            print(f'>> Pressing Key: {action['key']}')")
                script_lines.append(f"            page.keyboard.press('{action['key']}')")
            
            elif act == "screenshot":
                script_lines.append(f"            print(f'>> Taking screenshot: {action['filename']}')")
                script_lines.append(f"            page.screenshot(path='{action['filename']}')")
                
            elif act == "scroll":
                if action['direction'] == 'down':
                     script_lines.append("            print('>> Scrolling down')")
                     script_lines.append("            page.evaluate('window.scrollBy(0, window.innerHeight)')")
                else:
                     script_lines.append("            print('>> Scrolling up')")
                     script_lines.append("            page.evaluate('window.scrollBy(0, -window.innerHeight)')")

            elif act == "wait":
                script_lines.append(f"            print(f'>> Waiting for {action['seconds']} seconds')")
                script_lines.append(f"            page.wait_for_timeout({int(action['seconds']) * 1000})")
            
            elif act == "go_back":
                script_lines.append("            print('>> Navigating back')")
                script_lines.append("            page.go_back()")
                
            elif act == "go_forward":
                script_lines.append("            print('>> Navigating forward')")
                script_lines.append("            page.go_forward()")

            elif act == "error_if_empty":
                 # Custom logic for 'if no element on the cart then give error'
                 pass 

        script_lines.append("        except Exception as e:")
        script_lines.append("            print(f'>> ERROR: {e}', file=sys.stderr)")
        script_lines.append("        finally:")
        script_lines.append("            browser.close()")
        
        script_lines.append("")
        script_lines.append("if __name__ == '__main__':")
        script_lines.append("    run()")
        
        return "\n".join(script_lines)
