import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

const desktopOnly = (isMobile: boolean) => test.skip(isMobile, 'needs a mouse');
const phoneOnly = (isMobile: boolean) => test.skip(!isMobile, 'phone layout');

async function openMap(page: Page): Promise<{ map: Locator; box: { x: number; y: number; width: number; height: number } }> {
	await page.goto('./');
	const map = page.getByRole('application');
	await map.scrollIntoViewIfNeeded();
	return { map, box: (await map.boundingBox())! };
}

/** Hover across the upper right of the map until the description lists only some diagnoses. */
async function findLongList(page: Page, map: Locator, box: { x: number; y: number; width: number; height: number }) {
	const tip = map.locator('.tip');
	for (let fx = 0.55; fx < 0.9; fx += 0.02) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * 0.35);
		if ((await tip.count()) > 0 && /and \d+ more/.test(await tip.innerText())) return tip;
	}
	throw new Error('no dot with a shortened list of diagnoses found');
}

test('the opening is in the HTML before any script runs', async ({ browser }) => {
	const context = await browser.newContext({ javaScriptEnabled: false });
	const page = await context.newPage();
	await page.goto('./');
	await expect(page.getByRole('heading', { level: 1 })).toContainText('sort them into types');
	await expect(page.getByText(/This exact row of answers is shared by \d+ children/)).toBeVisible();
	await expect(page.getByText('Five lifelong conditions')).toBeVisible();
	await expect(page.getByText('Nothing here can tell you which group, if any, a particular child is in')).toBeVisible();
	await context.close();
});

test('the map draws and labels the groups', async ({ page }) => {
	const { map } = await openMap(page);
	await expect(map).toBeVisible();
	await expect(page.getByText('fewer diagnoses').first()).toBeVisible();
});

test('pointing at a dot describes its answers without a year or a verdict on the child', async ({ page, isMobile }) => {
	desktopOnly(isMobile);
	const { map, box } = await openMap(page);
	await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.55);
	const tip = map.locator('.tip');
	await expect(tip).toContainText("best fit for these answers");
	await expect(tip).not.toContainText(/20\d\d/);
});

test('a pinned description expands, and the keyboard reaches its buttons', async ({ page, isMobile }) => {
	desktopOnly(isMobile);
	const { map, box } = await openMap(page);
	const tip = await findLongList(page, map, box);
	await page.mouse.down();
	await page.mouse.up();
	await page.keyboard.press('Tab');
	await expect(tip.getByRole('button', { name: 'Show all' })).toBeFocused();
	await page.keyboard.press('Enter');
	await expect(tip).not.toContainText(/and \d+ more/);
	await tip.getByRole('button', { name: 'Close' }).click();
	await expect(tip).toHaveCount(0);
	await expect(map).toBeFocused();
});

test('the arrow keys move between children and speak only then', async ({ page, isMobile }) => {
	desktopOnly(isMobile);
	const { map, box } = await openMap(page);
	const readout = page.locator('[aria-live="polite"]');
	await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.55);
	await expect(readout).toBeEmpty();
	await map.focus();
	await page.keyboard.press('ArrowRight');
	await expect(readout).toContainText('best fit for these answers');
});

test('on a phone, a tap pins a dot below the map and nothing scrolls sideways', async ({ page, isMobile }) => {
	phoneOnly(isMobile);
	const { map, box } = await openMap(page);
	await map.dispatchEvent('pointerdown', { pointerType: 'touch', clientX: box.x + 10, clientY: box.y + 10 });
	await expect(page.locator('.tip')).toHaveCount(0);
	await page.touchscreen.tap(box.x + box.width * 0.6, box.y + box.height * 0.4);
	await expect(page.locator('.tip.static')).toBeVisible();
	const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
	expect(overflow).toBeLessThanOrEqual(0);
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
	await openMap(page);
	const results = await new AxeBuilder({ page }).analyze();
	expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
});
