# The project website

A static SvelteKit site that tells the analysis as a story, with the map of children as its main figure. It reads only the JSON files in `static/data/`, which `make site-data` writes from the pipeline's results; what those files may contain about a child is set out in `src/nsch_subtypes/site_data.py`.

```bash
cd site
npm ci
npm run dev          # http://localhost:5173
npm test             # unit tests (Vitest)
npm run test:e2e     # builds under the GitHub Pages subpath and runs the smoke tests (Playwright, axe)
npm run check        # type check
```

| Path | What it holds |
|---|---|
| `src/routes/` | The three pages: the story (`+page.svelte`), Explore and Methods. |
| `src/lib/components/ChildMap.svelte` | Every child as a dot on a canvas, with hover, tap and arrow-key reading. |
| `src/lib/describe.ts` | What the page may say about one dot: rarer combinations of answers are described only by their number of conditions and group probabilities. |
| `src/lib/groups.ts` | The four group colors (checked for color-vision deficiency) and condition labels. |
| `tests/smoke.spec.ts` | Loads, the map, hover, keyboard, navigation under the subpath, and an axe accessibility scan. |
