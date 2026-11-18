import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('the story loads, draws the map and labels the groups', async ({ page }) => {
	await page.goto('./');
	await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
	await expect(page.getByText('Nothing here describes or diagnoses an individual child')).toBeVisible();
	const map = page.getByRole('application');
	await expect(map).toBeVisible();
	await expect(map.getByText('fewer diagnoses')).toBeVisible();
});

test('pointing at the middle of the map describes a child without a year', async ({ page }) => {
	await page.goto('./');
	const map = page.getByRole('application');
	await map.scrollIntoViewIfNeeded();
	const box = (await map.boundingBox())!;
	await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.55);
	const tip = map.locator('.tip');
	await expect(tip).toBeVisible();
	await expect(tip).not.toContainText(/20\d\d/);
});

test('clicking a child with many conditions pins it and shows them all', async ({ page }) => {
	await page.goto('./');
	const map = page.getByRole('application');
	await map.scrollIntoViewIfNeeded();
	const box = (await map.boundingBox())!;
	const tip = map.locator('.tip');
	// Children with several conditions sit up and to the right; find one whose list is shortened.
	let found = false;
	for (let fx = 0.55; fx < 0.9 && !found; fx += 0.02) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * 0.35);
		found = (await tip.count()) > 0 && /and \d+ more/.test(await tip.innerText());
	}
	expect(found).toBe(true);
	await page.mouse.down();
	await page.mouse.up();
	await tip.getByRole('button', { name: 'Show all' }).click();
	await expect(tip).not.toContainText(/and \d+ more/);
	await tip.getByRole('button', { name: 'Close' }).click();
	await expect(tip).toHaveCount(0);
});

test('the arrow keys move between children', async ({ page }) => {
	await page.goto('./');
	const map = page.getByRole('application');
	await map.focus();
	await page.keyboard.press('ArrowRight');
	await expect(page.locator('#map-readout')).not.toBeEmpty();
});

test('every page is reachable under the subpath', async ({ page }) => {
	for (const [link, heading] of [
		['Explore', 'Explore'],
		['Methods and paper', 'Methods and paper'],
		['Story', 'Do autistic children']
	]) {
		await page.goto('./');
		await page.getByRole('navigation').getByRole('link', { name: link, exact: true }).click();
		await expect(page.getByRole('heading', { level: 1 })).toContainText(heading);
	}
});

test('the story has no detectable accessibility violations', async ({ page }) => {
	await page.goto('./');
	await expect(page.getByRole('application')).toBeVisible();
	const results = await new AxeBuilder({ page }).analyze();
	expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
});
