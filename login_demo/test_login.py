print("Script started")

from playwright.sync_api import sync_playwright
import os

def test_login():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Get absolute path of login.html
        file_path = os.path.abspath("login_demo/login.html")
        page.goto(f"file:///{file_path}")

        # Fill username and password
        page.fill("#username", "Dhikshitha")
        page.fill("#password", "12345")

        # Click login button
        page.click("button")

        # Verify success message
        page.wait_for_selector("#message")
        message = page.inner_text("#message")

        assert "Login Successful" in message

        print("Test Passed Successfully!")

        browser.close()

test_login()
