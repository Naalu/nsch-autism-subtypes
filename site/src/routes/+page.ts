import { loadData, type Context, type Model, type Numbers } from '#lib/data.ts';
import type { PageLoad } from './$types';

// The small files are read at build time, so the opening and its numbers are in the HTML.
// The map's per-child files are fetched in the browser.
export const load: PageLoad = async ({ fetch }) => {
	const [model, numbers, context] = await Promise.all([
		loadData<Model>('model.json', fetch),
		loadData<Numbers>('numbers.json', fetch),
		loadData<Context>('context.json', fetch)
	]);
	return { model, numbers, context };
};
