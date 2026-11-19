import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [
		sveltekit({
			preprocess: vitePreprocess(),
			// Every page is prerendered to plain HTML; 404.html lets GitHub Pages serve unknown paths.
			adapter: adapter({ fallback: '404.html' }),

			// GitHub Pages serves the site under /<repository>/; the deploy workflow sets BASE_PATH.
			paths: { base: (process.env.BASE_PATH ?? '') as '' | `/${string}` }
		})
	],
	test: { include: ['src/**/*.test.ts'] }
});
