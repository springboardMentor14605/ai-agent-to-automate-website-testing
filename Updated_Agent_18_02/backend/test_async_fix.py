"""
Test script to verify async Playwright works on Windows
Run this to test if the event loop policy fix is working
"""
import asyncio
import platform
from playwright.async_api import async_playwright

# Apply Windows fix
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def test_playwright():
    """Simple test to verify Playwright async works"""
    print(f"Platform: {platform.system()}")
    print(f"Event Loop Policy: {asyncio.get_event_loop_policy()}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto('https://www.google.com')
        title = await page.title()
        
        print(f"✓ Successfully loaded: {title}")
        
        await browser.close()
        print("✓ Playwright async test passed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_playwright())
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
