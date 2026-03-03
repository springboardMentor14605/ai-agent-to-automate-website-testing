import os
import pytest
from playwright.sync_api import sync_playwright, expect

LOGIN_PAGE = "file:///C:/Users/kabha/OneDrive/Desktop/Infosys/INFS_SPRNBRD/login.html"
SCREENSHOT_DIR = "screenshots"


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
    page = browser.new_page()
    request.node.page = page
    yield page
    page.close()


def test_positive_login(page):
    page.goto(LOGIN_PAGE)
    page.fill("#username", "testuser")
    page.fill("#password", "testpassword")
    page.click("#loginBtn")

    expect(page.locator("#successMessage")).to_have_text("Login Successful!")


def test_negative_login(page):
    page.goto(LOGIN_PAGE)
    page.fill("#username", "")
    page.fill("#password", "")
    page.click("#loginBtn")

    expect(page.locator("#errorMessage")).to_have_text("WRONG EXPECTED TEXT")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if rep.failed and hasattr(item, "page"):
        page = item.page
        if not page.is_closed():
            page.screenshot(
                path=os.path.join(
                    SCREENSHOT_DIR, f"{item.name}_{rep.when}.png"
                ),
                full_page=True
            )
