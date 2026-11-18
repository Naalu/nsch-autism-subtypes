import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		// Every page is prerendered to plain HTML; 404.html lets GitHub Pages serve unknown paths.
		adapter: adapter({ fallback: '404.html' }),
		// GitHub Pages serves the site under /<repository>/; the deploy workflow sets BASE_PATH.
		paths: { base: process.env.BASE_PATH ?? '' }
	}
};

export default config;
