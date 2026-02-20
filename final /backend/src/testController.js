import openai from "./openaiClient.js";
import { runPlaywrightTest } from "./playwrightService.js";

export async function runWebsiteTest({ url }) {
  // Ask AI to generate steps
  const prompt = `
    Generate a simple test plan for the site ${url}.
    Output an array of steps like:
    [{ action: "click", selector: "..."},
     { action: "type", selector: "...", value: "..."}]
  `;

  const completion = await openai.responses.create({
    model: "gpt-4o",
    input: prompt,
    max_output_tokens: 500
  });

  const stepsJson = completion.output_text.trim();

  let steps = [];
  try {
    steps = JSON.parse(stepsJson);
  } catch (e) {
    throw new Error("AI did not return valid steps");
  }

  // Run the steps in Playwright
  const result = await runPlaywrightTest(url, steps);
  return { steps, result };
}