from playwright.sync_api import sync_playwright, expect

def test_login():
    with sync_playwright() as p:
        # Launch browser (set headless=False to SEE the automation)
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Open the login page
        # Use file:// if it's a local HTML file
        page.goto("file:///C:/Users/kabha/OneDrive/Desktop/Infosys/INFS_SPRNBRD/login.html")
        # OR for hosted page:
        # page.goto("http://localhost:5500/login.html")

        # Simulate user typing username
        page.fill("#username", "testuser")

        # Simulate user typing password
        page.fill("#password", "testpassword")

        # Click the login button
        page.click("#loginBtn")

        # Verify success message appears
        success_message = page.locator("#successMessage")
        expect(success_message).to_be_visible()
        expect(success_message).to_have_text("Login Successful!")

        print("✅ Login automation test passed!")

        browser.close()


if __name__ == "__main__":
    test_login()

