<!-- The four group probabilities for one set of answers, as one bar, with the larger parts labeled. -->
<script lang="ts">
	import type { Group } from '$lib/data';
	import { GROUP_COLORS, percent } from '$lib/groups';

	let { probabilities, groups, labelMin = 0.08 }: { probabilities: number[]; groups: Group[]; labelMin?: number } = $props();
</script>

<div class="bar" role="img" aria-label={groups.map((g, i) => `${g.short_name} ${percent(probabilities[i])}`).join(', ')}>
	{#each probabilities as p, i}
		{#if p > 0.004}
			<span class="part" style:flex-grow={p} style:background={GROUP_COLORS[i]}></span>
		{/if}
	{/each}
</div>
<ul class="labels" aria-hidden="true">
	{#each probabilities as p, i}
		{#if p >= labelMin}
			<li><span class="key" style:background={GROUP_COLORS[i]}></span>{groups[i].short_name} <span class="num">{percent(p)}</span></li>
		{/if}
	{/each}
</ul>

<style>
	.bar {
		display: flex;
		gap: 2px;
		height: 0.9rem;
		border-radius: 3px;
		overflow: hidden;
	}
	.part {
		flex-basis: 0;
	}
	.labels {
		list-style: none;
		padding: 0;
		margin: 0.45rem 0 0;
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem 1.25rem;
		font-size: var(--step--1);
	}
	.key {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		margin-right: 0.4rem;
		vertical-align: -0.05em;
	}
</style>
