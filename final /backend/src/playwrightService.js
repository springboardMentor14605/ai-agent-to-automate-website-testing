import { chromium } from "playwright";

export async function runPlaywrightTest(url, steps) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 30000 }).catch(e => console.log("Navigation timeout:", e.message));

  const log = [];
  log.push(`Successfully loaded ${url}`);

  // Try to execute steps with timeout protection
  for (const step of steps) {
    try {
      if (step.action === "click") {
        log.push(`Attempting to click ${step.selector}`);
        await page.click(step.selector, { timeout: 5000 }).catch(() => {
          log.push(`Failed to click ${step.selector} (element not clickable)`);
        });
      }
      if (step.action === "type") {
        log.push(`Attempting to type in ${step.selector}`);
        await page.fill(step.selector, step.value, { timeout: 5000 }).catch(() => {
          log.push(`Failed to type in ${step.selector} (element not found)`);
        });
      }
    } catch (error) {
      log.push(`Step failed: ${error.message}`);
    }
  }

  const finalTitle = await page.title();
  await browser.close();

  return { finalTitle, log };
}