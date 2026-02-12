
const { test, expect } = require('@playwright/test');

test.describe('Practice Test Automation - Login Page', () => {

    // This runs before every test case
    test.beforeEach(async ({ page }) => {
        await page.goto('https://practicetestautomation.com/practice-test-login/');
    });

    test('Successful Login Test', async ({ page }) => {
        // 1. Fill in the username
        await page.fill('#username', 'student');

        // 2. Fill in the password
        await page.fill('#password', 'Password123');

        // 3. Click the Submit button
        await page.click('#submit');

        // 4. Assertions
        // Verify the URL has changed to the success page
        await expect(page).toHaveURL(/logged-in-successfully/);

        // Verify the success message header is visible and contains correct text
        const successHeader = page.locator('.post-title');
        await expect(successHeader).toBeVisible();
        await expect(successHeader).toHaveText('Logged In Successfully');

        // Verify the Logout button is present
        const logoutButton = page.locator('text=Log out');
        await expect(logoutButton).toBeVisible();
    });

    test('Invalid Username Login Test', async ({ page }) => {
        // 1. Fill in an incorrect username
        await page.fill('#username', 'incorrectUser');

        // 2. Fill in the password
        await page.fill('#password', 'Password123');

        // 3. Click Submit
        await page.click('#submit');

        // 4. Assertions
        // Verify error message is displayed
        const errorMessage = page.locator('#error');
        await expect(errorMessage).toBeVisible();
        await expect(errorMessage).toHaveText(/Your username is invalid!/);
    });

    test('Invalid Password Login Test', async ({ page }) => {
        // 1. Fill in the correct username
        await page.fill('#username', 'student');

        // 2. Fill in an incorrect password
        await page.fill('#password', 'wrongPassword');

        // 3. Click Submit
        await page.click('#submit');

        // 4. Assertions
        const errorMessage = page.locator('#error');
        await expect(errorMessage).toBeVisible();
        await expect(errorMessage).toHaveText(/Your password is invalid!/);
    });

});
