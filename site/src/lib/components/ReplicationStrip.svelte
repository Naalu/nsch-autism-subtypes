<!--
	The replication checks, real against made up. One row per check on a shared agreement
	scale: five open circles for the five made-up datasets with no groups, five filled
	squares for the five made-up datasets with four separate groups, and a bar for the real
	children. Shapes, not colors, tell them apart: the group colors mean groups only.
-->
<script lang="ts">
	import type { CheckRow } from '#lib/evidence.ts';

	let { rows }: { rows: CheckRow[] } = $props();

	const W = 420;
	const H = 40;
	const PAD = 10;
	const MIN = 0.4;
	const MAX = 1.0;
	const TICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
	const x = (v: number) => PAD + ((v - MIN) / (MAX - MIN)) * (W - 2 * PAD);
	const fmt = (v: number) => v.toFixed(2);
</script>

<figure class="strip">
	<ul class="legend">
		<li><svg viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="4.5" class="none" /></svg>Made-up children, no groups (5 sets)</li>
		<li><svg viewBox="0 0 14 14" aria-hidden="true"><rect x="3" y="3" width="8" height="8" class="four" /></svg>Made-up children, four separate groups (5 sets)</li>
		<li><svg viewBox="0 0 14 14" aria-hidden="true"><rect x="5.5" y="0" width="3" height="14" class="real" /></svg>The real children</li>
	</ul>

	<div class="axis-row">
		<span></span>
		<svg viewBox="0 0 {W} 18" aria-hidden="true">
			{#each TICKS as t}
				<text x={x(t)} y="12" text-anchor="middle" class="tick">{t.toFixed(1)}</text>
			{/each}
		</svg>
	</div>

	{#each rows as row}
		<div class="row">
			<p class="label">{row.label}</p>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label="{row.label}: real children {fmt(row.real)}; no groups {fmt(Math.min(...row.noGroups))} to {fmt(Math.max(...row.noGroups))}; four separate groups {fmt(Math.min(...row.fourGroups))} to {fmt(Math.max(...row.fourGroups))}">
				{#each TICKS as t}
					<line x1={x(t)} x2={x(t)} y1="2" y2={H - 2} class="grid" />
				{/each}
				{#each row.noGroups as v}
					<circle cx={x(v)} cy="13" r="4.5" class="none"><title>No groups: {fmt(v)}</title></circle>
				{/each}
				{#each row.fourGroups as v}
					<rect x={x(v) - 4} y="23" width="8" height="8" class="four"><title>Four separate groups: {fmt(v)}</title></rect>
				{/each}
				<rect x={x(row.real) - 1.5} y="2" width="3" height={H - 4} class="real"><title>Real children: {fmt(row.real)}</title></rect>
				<text x={x(row.real) + (row.real > 0.9 ? -6 : 6)} y="11" text-anchor={row.real > 0.9 ? 'end' : 'start'} class="value">{fmt(row.real)}</text>
			</svg>
		</div>
	{/each}
	<p class="scale-note">Agreement between two sortings of the same children: 0 is no better than chance, 1 is identical.</p>

	<details>
		<summary>The values as a table</summary>
		<table>
			<thead>
				<tr><th scope="col">Check</th><th scope="col">Real</th><th scope="col">No groups</th><th scope="col">Four separate groups</th></tr>
			</thead>
			<tbody>
				{#each rows as row}
					<tr>
						<th scope="row">{row.label}</th>
						<td class="num">{fmt(row.real)}</td>
						<td class="num">{row.noGroups.map(fmt).join(', ')}</td>
						<td class="num">{row.fourGroups.map(fmt).join(', ')}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</details>
</figure>

<style>
	.strip {
		margin: 1.5rem 0 0;
		max-width: 52rem;
		font-size: var(--step--1);
	}
	.legend {
		list-style: none;
		padding: 0;
		margin: 0 0 0.75rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1.5rem;
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.legend svg {
		width: 14px;
		height: 14px;
	}
	.axis-row,
	.row {
		display: grid;
		grid-template-columns: 15rem minmax(0, 1fr);
		gap: 1rem;
		align-items: center;
	}
	@media (max-width: 40rem) {
		.axis-row,
		.row {
			grid-template-columns: minmax(0, 1fr);
			gap: 0.1rem;
		}
		.axis-row span {
			display: none;
		}
		.row {
			margin-top: 0.6rem;
		}
	}
	.label {
		margin: 0;
		line-height: 1.3;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font);
	}
	.none {
		fill: #fff;
		stroke: #6b7280;
		stroke-width: 1.5;
	}
	.four {
		fill: #4c5566;
	}
	.real {
		fill: var(--ink);
	}
	.value {
		paint-order: stroke;
		stroke: #fff;
		stroke-width: 4px;
		stroke-linejoin: round;
		font-size: 12px;
		font-weight: 600;
		fill: var(--ink);
		font-family: var(--font);
	}
	.scale-note {
		color: var(--ink-2);
		margin-top: 0.6rem;
	}
	details {
		margin-top: 0.75rem;
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
		vertical-align: top;
	}
	thead th {
		font-weight: 600;
	}
</style>
