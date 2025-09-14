// @ts-check
/**
 * Playwright E2E test for file upload and result display
 * Assumes dev server is running on http://localhost:3000
 */
const { test, expect } = require('@playwright/test');
const path = require('path');

test('User uploads file and sees result', async ({ page }) => {
  // Go to homepage
  await page.goto('http://localhost:3000');

  // Find file input and upload a file
  const filePath = path.resolve(__dirname, '../../../backend/data/HARSHA Kota Resume.pdf');
  const input = await page.$('input[type="file"]');
  expect(input).not.toBeNull();
  await input.setInputFiles(filePath);

  // Click upload button if present
  const uploadBtn = await page.$('button[type="submit"], button:has-text("Upload")');
  if (uploadBtn) {
    await uploadBtn.click();
  }

  // Wait for result (simulate analytics result display)
  await page.waitForSelector('.result, .analysis, .success, .output', { timeout: 10000 });
  const result = await page.$('.result, .analysis, .success, .output');
  expect(result).not.toBeNull();
  const text = await result.textContent();
  expect(text).not.toBe('');
});
