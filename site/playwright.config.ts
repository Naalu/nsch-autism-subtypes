import { defineConfig, devices } from '@playwright/test';

// The smoke tests run against the built site under the same subpath GitHub Pages uses,
// on a desktop window and on a phone-sized touch screen.
const BASE = '/nsch-autism-subtypes';

export default defineConfig({
	testDir: 'tests',
	use: { baseURL: `http://localhost:4173${BASE}/` },
	projects: [
		{ name: 'desktop', use: { viewport: { width: 1280, height: 860 } } },
		{ name: 'phone', use: { ...devices['iPhone 13'], browserName: 'chromium' } }
	],
	webServer: {
		command: 'npm run build && npx vite preview --port 4173 --strictPort',
		env: { BASE_PATH: BASE },
		port: 4173,
		reuseExistingServer: false,
		timeout: 120_000
	}
});
