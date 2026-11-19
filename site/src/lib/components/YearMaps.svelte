<!--
	The four groups year by year: the children of one survey year as patches of the map,
	colored by the most common group in each patch, with a slider, a play button and a
	thumbnail per year; beside it, each group's weighted share by year. Nothing moves
	unless the reader presses Play.
-->
<script lang="ts">
	import type { Model, Years } from '#lib/data.ts';
	import { dominant, sharesByGroup } from '#lib/evidence.ts';
	import { GROUP_COLORS, percent } from '#lib/groups.ts';

	let { years, model }: { years: Years; model: Model } = $props();

	const LIMIT = 3.75;
	const MAP = 300;
	const STEP_MS = 900;

	let index = $state(0);
	let playing = $state(false);

	const maps = $derived(years.maps);
	const current = $derived(maps[index]);
	const maxTotal = $derived(Math.max(...maps.flatMap((m) => m.cells.map((c) => c.counts.reduce((a, b) => a + b, 0)))));
	const cellSize = $derived((years.cell / (2 * LIMIT)) * MAP);
	const series = $derived(sharesByGroup(years, model.classes.map((g) => g.name)));

	const px = (v: number) => ((v + LIMIT) / (2 * LIMIT)) * MAP;
	const py = (v: number) => MAP - px(v);

	function patches(cells: Years['maps'][number]['cells']) {
		return cells.map((c) => {
			const d = dominant(c.counts);
			return {
				x: px(c.x) - cellSize / 2,
				y: py(c.y) - cellSize / 2,
				color: GROUP_COLORS[d.group],
				opacity: 0.15 + 0.85 * Math.sqrt(d.total / maxTotal),
				title: `${d.total} children; most common: ${model.classes[d.group].short_name} (${percent(d.share)})`
			};
		});
	}

	$effect(() => {
		if (!playing) return;
		const timer = setInterval(() => {
			if (index >= maps.length - 1) {
				playing = false;
				return;
			}
			index += 1;
		}, STEP_MS);
		return () => clearInterval(timer);
	});

	function togglePlay() {
		if (!playing && index >= maps.length - 1) index = 0;
		playing = !playing;
	}

	// The shares chart.
	const W = 360;
	const H = 230;
	const M = { top: 10, right: 12, bottom: 26, left: 34 };
	const yMax = 0.6;
	const firstYear = $derived(maps[0].year);
	const lastYear = $derived(maps[maps.length - 1].year);
	const sxYear = (year: number) => M.left + ((year - firstYear) / (lastYear - firstYear)) * (W - M.left - M.right);
	const syShare = (share: number) => M.top + (1 - share / yMax) * (H - M.top - M.bottom);
	const line = (points: { year: number; share: number }[]) => points.map((p, i) => `${i ? 'L' : 'M'}${sxYear(p.year)},${syShare(p.share)}`).join('');
	const band = (points: { year: number; low: number; high: number }[]) =>
		points.map((p, i) => `${i ? 'L' : 'M'}${sxYear(p.year)},${syShare(p.high)}`).join('') +
		[...points].reverse().map((p) => `L${sxYear(p.year)},${syShare(p.low)}`).join('') +
		'Z';
	const shareNow = (g: number) => series[g].find((p) => p.year === current.year);
</script>

<figure class="years">
	<div class="panels">
		<div class="map-panel">
			<div class="controls">
				<button type="button" class="play" onclick={togglePlay} aria-pressed={playing}>{playing ? 'Pause' : 'Play'}</button>
				<label>
					<span class="visually-hidden">Survey year</span>
					<input type="range" min="0" max={maps.length - 1} step="1" bind:value={index} onpointerdown={() => (playing = false)} />
				</label>
				<output class="year num" aria-live="polite">{current.year}</output>
			</div>
			<svg viewBox="0 0 {MAP} {MAP}" role="img" aria-label="Map of the {current.n} children surveyed in {current.year}, by most common group in each patch">
				<rect width={MAP} height={MAP} fill="#fff" />
				{#each patches(current.cells) as p}
					<rect x={p.x} y={p.y} width={cellSize - 1} height={cellSize - 1} fill={p.color} fill-opacity={p.opacity} rx="2">
						<title>{p.title}</title>
					</rect>
				{/each}
			</svg>
			<p class="note">{current.n.toLocaleString('en-US')} children in {current.year}</p>
		</div>

		<div class="chart-panel">
			<p class="chart-title">Each group's share by survey year</p>
			<ul class="legend">
				{#each model.classes as g, i}
					{@const now = shareNow(i)}
					<li><span class="key" style:background={GROUP_COLORS[i]}></span>{g.short_name} <span class="num">{now ? percent(now.share) : ''}</span></li>
				{/each}
			</ul>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label="Each group's weighted share of autistic children by survey year, roughly steady from {firstYear} to {lastYear}">
				{#each [0, 0.2, 0.4, 0.6] as tick}
					<line x1={M.left} x2={W - M.right} y1={syShare(tick)} y2={syShare(tick)} class="grid" />
					<text x={M.left - 6} y={syShare(tick)} class="tick" text-anchor="end" dominant-baseline="middle">{percent(tick)}</text>
				{/each}
				{#each maps as m}
					{#if (m.year - firstYear) % 2 === 0}
						<text x={sxYear(m.year)} y={H - 8} class="tick" text-anchor="middle">{m.year}</text>
					{/if}
				{/each}
				<line x1={sxYear(current.year)} x2={sxYear(current.year)} y1={M.top} y2={H - M.bottom} class="now" />
				{#each series as points, g}
					<path d={band(points)} fill={GROUP_COLORS[g]} fill-opacity="0.13" />
				{/each}
				{#each series as points, g}
					<path d={line(points)} fill="none" stroke={GROUP_COLORS[g]} stroke-width="2" />
					{@const now = points.find((p) => p.year === current.year)}
					{#if now}
						<circle cx={sxYear(now.year)} cy={syShare(now.share)} r="4" fill={GROUP_COLORS[g]} stroke="#fff" stroke-width="2">
							<title>{model.classes[g].short_name}, {current.year}: {percent(now.share)} ({percent(now.low)} to {percent(now.high)})</title>
						</circle>
					{/if}
				{/each}
			</svg>
		</div>
	</div>

	<div class="thumbs" role="group" aria-label="Pick a survey year">
		{#each maps as m, i}
			<button type="button" aria-pressed={i === index} onclick={() => ((index = i), (playing = false))}>
				<svg viewBox="0 0 {MAP} {MAP}" aria-hidden="true">
					<rect width={MAP} height={MAP} fill="#fff" />
					{#each patches(m.cells) as p}
						<rect x={p.x} y={p.y} width={cellSize - 1} height={cellSize - 1} fill={p.color} fill-opacity={p.opacity} />
					{/each}
				</svg>
				<span class="num">{m.year}</span>
			</button>
		{/each}
	</div>

	<details>
		<summary>The shares as a table</summary>
		<table>
			<thead>
				<tr><th scope="col">Year</th>{#each model.classes as g}<th scope="col">{g.short_name}</th>{/each}</tr>
			</thead>
			<tbody>
				{#each maps as m}
					<tr>
						<th scope="row" class="num">{m.year}</th>
						{#each series as points}
							{@const p = points.find((q) => q.year === m.year)}
							<td class="num">{p ? `${percent(p.share)} (${percent(p.low)} to ${percent(p.high)})` : ''}</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</details>
</figure>

<style>
	.years {
		margin: 1.5rem 0 0;
	}
	.panels {
		display: grid;
		grid-template-columns: minmax(0, 20rem) minmax(0, 24rem);
		gap: 2rem 3rem;
		align-items: start;
	}
	@media (max-width: 46rem) {
		.panels {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.map-panel svg {
		outline: 1px solid var(--rule);
		border-radius: 4px;
	}
	.controls {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.6rem;
		font-size: var(--step--1);
	}
	.controls label {
		flex: 1;
		display: flex;
	}
	input[type='range'] {
		width: 100%;
		accent-color: var(--ink);
	}
	.year {
		font-weight: 600;
		font-size: var(--step-1);
		min-width: 3.2rem;
		text-align: right;
	}
	button {
		font: inherit;
		color: var(--ink);
		background: #fff;
		border: 1px solid var(--rule);
		border-radius: 999px;
		padding: 0.25rem 0.9rem;
		cursor: pointer;
	}
	.play[aria-pressed='true'] {
		background: var(--ink);
		color: #fff;
		border-color: var(--ink);
	}
	.note,
	.chart-title {
		font-size: var(--step--1);
		color: var(--ink-2);
		margin: 0.4rem 0 0;
	}
	.chart-title {
		color: var(--ink);
		font-weight: 600;
		margin: 0 0 0.4rem;
	}
	.legend {
		list-style: none;
		padding: 0;
		margin: 0 0 0.5rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.2rem 1rem;
		font-size: var(--step--1);
	}
	.key {
		display: inline-block;
		width: 0.65rem;
		height: 0.65rem;
		border-radius: 50%;
		margin-right: 0.35rem;
	}
	.grid {
		stroke: var(--rule);
	}
	.now {
		stroke: var(--ink);
		stroke-dasharray: 3 3;
	}
	.tick {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font);
	}
	.thumbs {
		display: grid;
		grid-template-columns: repeat(9, minmax(0, 1fr));
		gap: 0.5rem;
		margin-top: 1.5rem;
		max-width: 46rem;
	}
	@media (max-width: 46rem) {
		.thumbs {
			grid-template-columns: repeat(5, minmax(0, 1fr));
		}
	}
	.thumbs button {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.2rem;
		padding: 0.3rem;
		border-radius: 4px;
		font-size: 0.75rem;
	}
	.thumbs button[aria-pressed='true'] {
		border-color: var(--ink);
		box-shadow: inset 0 0 0 1px var(--ink);
	}
	details {
		margin-top: 1rem;
		font-size: var(--step--1);
	}
	summary {
		cursor: pointer;
		color: var(--ink-2);
	}
	table {
		border-collapse: collapse;
		margin-top: 0.5rem;
	}
	th,
	td {
		text-align: left;
		padding: 0.15rem 0.75rem 0.15rem 0;
		font-weight: 400;
	}
	thead th {
		font-weight: 600;
	}
</style>
