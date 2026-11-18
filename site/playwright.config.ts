import { defineConfig } from '@playwright/test';

// The smoke tests run against the built site under the same subpath GitHub Pages uses.
const BASE = '/nsch-autism-subtypes';

export default defineConfig({
	testDir: 'tests',
	use: { baseURL: `http://localhost:4173${BASE}/` },
	webServer: {
		command: `npm run build && npx vite preview --port 4173 --strictPort`,
		env: { BASE_PATH: BASE },
		port: 4173,
		reuseExistingServer: false,
		timeout: 120_000
	}
});
