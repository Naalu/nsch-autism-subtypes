<!--
	Three sets of children, unlabeled: the real survey, made-up children from four separate
	groups, and made-up children with no groups. The reader guesses which has real groups;
	the answer brings in a fixed second row drawn from the estimated distribution, with its
	warning on the figure. There is no free toggle: the comparison is always side by side.
-->
<script lang="ts">
	import type { Clouds } from '#lib/data.ts';
	import CloudCanvas from './CloudCanvas.svelte';

	let { clouds }: { clouds: Clouds } = $props();

	const SETS = [
		{ key: 'continuum', letter: 'A', name: 'Made up: no groups' },
		{ key: 'real', letter: 'B', name: 'The real survey' },
		{ key: 'classes', letter: 'C', name: 'Made up: four separate groups' }
	] as const;
	const ANSWER = 'C';

	let guess = $state<string | null>(null);

	const feedback = $derived(
		guess === null
			? ''
			: guess === ANSWER
				? 'Right: C was made up from four separate groups. Drawn this way, though, all three look like one round cloud, so the picture alone could not have told you.'
				: guess === 'unsure'
					? 'Neither could we, drawn this way: all three look like one round cloud. C was made up from four separate groups.'
					: `C was the one made up from four separate groups, not ${guess}. Drawn this way, all three look like one round cloud.`
	);
</script>

<div class="guess">
	<div class="row">
		{#each SETS as set}
			<figure>
				<CloudCanvas cloud={clouds[set.key]} label="Set {set.letter}{guess ? `: ${set.name}` : ''}, drawn assuming a bell curve" />
				<figcaption><strong>{set.letter}</strong>{#if guess}<span class="name">{set.name}</span>{/if}</figcaption>
			</figure>
		{/each}
	</div>

	{#if guess === null}
		<div class="choices" role="group" aria-label="Which set has real groups?">
			<span>Which one has real groups?</span>
			{#each SETS as set}
				<button type="button" onclick={() => (guess = set.letter)}>{set.letter}</button>
			{/each}
			<button type="button" onclick={() => (guess = 'unsure')}>I can't tell</button>
		</div>
	{:else}
		<p class="feedback" role="status">{feedback}</p>
		<p class="row-label">The same children, each drawn from a distribution estimated from the data</p>
		<div class="row">
			{#each SETS as set}
				<figure>
					<CloudCanvas cloud={clouds[set.key]} sharp label="{set.name}, drawn from the estimated distribution" />
					<figcaption><strong>{set.letter}</strong><span class="name">{set.name}</span></figcaption>
				</figure>
			{/each}
		</div>
		<p class="warning">
			This second drawing follows the data closely, including its noise, so it exaggerates lumps. It shows the direction of the
			difference; the measurement below gives its size.
		</p>
	{/if}
</div>

<style>
	.guess {
		margin-top: 1.5rem;
		max-width: 46rem;
	}
	.row {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 1rem;
	}
	figure {
		margin: 0;
	}
	figcaption {
		margin-top: 0.35rem;
		font-size: var(--step--1);
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
	}
	.name {
		color: var(--ink-2);
	}
	.choices {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.5rem;
		margin-top: 1rem;
		font-size: var(--step--1);
	}
	.choices span {
		font-weight: 600;
		margin-right: 0.25rem;
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
	button:hover {
		border-color: var(--ink);
	}
	.feedback {
		margin: 1rem 0 1.5rem;
		font-weight: 600;
	}
	.row-label,
	.warning {
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.row-label {
		margin-bottom: 0.5rem;
	}
	.warning {
		margin-top: 0.75rem;
		border-left: 2px solid var(--ink);
		padding-left: 0.75rem;
	}
</style>
