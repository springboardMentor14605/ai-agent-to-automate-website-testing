import os
import pytest
import sys
from playwright.sync_api import sync_playwright, expect


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
LOGIN_PAGE = f"file:///{os.path.join(PROJECT_ROOT, 'login.html').replace(os.sep, '/')}"


@pytest.fixture(scope="session")
def playwright_instance():
    p = sync_playwright().start()
    yield p
    p.stop()

@pytest.fixture(scope="session")
def browser(playwright_instance):
    browser = playwright_instance.chromium.launch(headless=False)
    yield browser
    browser.close()

@pytest.fixture(scope="function")
def page(request, browser):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    
    context = browser.new_context()
    page = context.new_page()
    
    yield page
    

    node = request.node
    if getattr(node, "rep_call", None) and node.rep_call.failed:
        
        safe_name = node.name.replace("::", "_").replace(".py", "")
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"{safe_name}_failed.png")
        
        try:
            if not page.is_closed():
                page.screenshot(path=screenshot_path, full_page=True)
                
                print(f"\n[SCREENSHOT CAPTURED] Saved to: {screenshot_path}")
                sys.stderr.write(f"\n[SCREENSHOT CAPTURED] Saved to: {screenshot_path}\n")
        except Exception as e:
            print(f"\n[SCREENSHOT ERROR] Could not take screenshot: {e}")


    page.close()
    context.close()


def test_positive_login(page):
    """This test should PASS."""
    page.goto(LOGIN_PAGE)
    page.fill("#username", "testuser")
    page.fill("#password", "testpassword")
    page.click("#loginBtn")
    expect(page.locator("#successMessage")).to_have_text("Login Successful!")

def test_negative_login(page):
    """Test login with empty credentials — should display an error."""
    page.goto(LOGIN_PAGE)
    page.fill("#username", "") 
    page.fill("#password", "")
    page.click("#loginBtn")

    expect(page.locator("#errorMessage")).to_have_text("Invalid credentials")