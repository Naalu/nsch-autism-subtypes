<!-- The four-group model's fit for one set of answers: one bar, best fit first, larger parts labeled. -->
<script lang="ts">
	import type { Group } from '#lib/data.ts';
	import { GROUP_COLORS, percent } from '#lib/groups.ts';

	let { probabilities, groups, labelMin = 0.08 }: { probabilities: number[]; groups: Group[]; labelMin?: number } = $props();

	const parts = $derived(
		probabilities
			.map((p, group) => ({ p, group, name: groups[group].short_name, color: GROUP_COLORS[group] }))
			.filter((part) => part.p > 0.004)
			.sort((a, b) => b.p - a.p)
	);
</script>

<div class="bar" role="img" aria-label="Best fit for these answers: {parts.map((part) => `${part.name} ${percent(part.p)}`).join(', ')}">
	{#each parts as part}
		<span class="part" style:flex-grow={part.p} style:background={part.color}></span>
	{/each}
</div>
<ul class="labels" aria-hidden="true">
	{#each parts.filter((part) => part.p >= labelMin) as part}
		<li><span class="key" style:background={part.color}></span>{part.name} <span class="num">{percent(part.p)}</span></li>
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
