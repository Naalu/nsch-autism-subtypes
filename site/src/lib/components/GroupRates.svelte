<!--
	What each group contains: for the eight diagnoses that name the two scales, the share of
	children in each group who have it. A table, so it reads the same to a screen reader.
-->
<script lang="ts">
	import type { Model } from '#lib/data.ts';
	import { CONDITION_FAMILIES, CONDITION_SHORT, GROUP_COLORS, percent } from '#lib/groups.ts';

	let { model }: { model: Model } = $props();

	const families = $derived(
		CONDITION_FAMILIES.slice(0, 2).map((family) => ({
			name: family.name,
			rows: family.flags.map((flag) => {
				const j = model.conditions.findIndex((c) => c.flag === flag);
				return { flag, name: CONDITION_SHORT[flag], rates: model.classes.map((g) => g.rates[j]) };
			})
		}))
	);
</script>

<div class="scroll">
<table>
	<caption>Share of each group with each diagnosis</caption>
	<thead>
		<tr>
			<td></td>
			{#each model.classes as g, i}
				<th scope="col"><span class="key" style:background={GROUP_COLORS[i]}></span>{g.short_name}</th>
			{/each}
		</tr>
	</thead>
	{#each families as family}
		<tbody>
			<tr class="family"><th scope="rowgroup" colspan={model.classes.length + 1}>{family.name}</th></tr>
			{#each family.rows as row}
				<tr>
					<th scope="row">{row.name}</th>
					{#each row.rates as rate, i}
						<td>
							<span class="track"><span class="fill" style:width="{100 * rate}%" style:background={GROUP_COLORS[i]}></span></span>
							<span class="num">{percent(rate)}</span>
						</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	{/each}
</table>
</div>

<style>
	.scroll {
		overflow-x: auto;
		max-width: 46rem;
	}
	table {
		width: 100%;
		max-width: 46rem;
		border-collapse: collapse;
		font-size: var(--step--1);
	}
	caption {
		text-align: left;
		font-weight: 600;
		padding-bottom: 0.5rem;
	}
	thead th {
		font-weight: 600;
		text-align: left;
		vertical-align: bottom;
		padding: 0 0.5rem 0.4rem;
		line-height: 1.25;
	}
	.family th {
		text-align: left;
		color: var(--ink-2);
		font-weight: 400;
		padding: 0.9rem 0 0.25rem;
	}
	tbody th[scope='row'] {
		text-align: left;
		font-weight: 400;
		padding: 0.2rem 0.75rem 0.2rem 0;
		white-space: nowrap;
	}
	td {
		padding: 0.2rem 0.5rem;
		white-space: nowrap;
	}
	.track {
		display: inline-block;
		width: 3.5rem;
		height: 0.55rem;
		background: #e6e9ee;
		border-radius: 2px;
		overflow: hidden;
		vertical-align: middle;
		margin-right: 0.4rem;
	}
	.fill {
		display: block;
		height: 100%;
	}
	.key {
		display: inline-block;
		width: 0.6rem;
		height: 0.6rem;
		border-radius: 50%;
		margin-right: 0.35rem;
	}
	@media (max-width: 40rem) {
		.track {
			display: none;
		}
		table {
			table-layout: fixed;
		}
		thead td {
			width: 28%;
		}
		thead th {
			font-size: 0.75rem;
			hyphens: auto;
			overflow-wrap: break-word;
		}
		thead th,
		td {
			padding-inline: 0.25rem;
		}
		tbody th[scope='row'] {
			white-space: normal;
		}
	}
</style>
