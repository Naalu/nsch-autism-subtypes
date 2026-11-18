<script lang="ts">
	import { onMount } from 'svelte';
	import ChildMap from '#lib/components/ChildMap.svelte';
	import GroupRates from '#lib/components/GroupRates.svelte';
	import ProbabilityBar from '#lib/components/ProbabilityBar.svelte';
	import ProfileRow from '#lib/components/ProfileRow.svelte';
	import { loadData, type Children, type Profiles } from '#lib/data.ts';
	import { percent } from '#lib/groups.ts';

	let { data } = $props();

	const TITLE = "Do autistic children's other diagnoses sort them into types?";
	const AGELINK = 'https://doi.org/10.1016/j.cpr.2021.102033';
	const LITMAN = 'https://doi.org/10.1038/s41588-025-02224-z';

	const n = $derived(data.numbers.children.value.toLocaleString('en-US'));
	const example = $derived(data.context.example);
	const [first, second] = $derived(
		example.class_probabilities
			.map((p, group) => ({ p, group, name: data.model.classes[group].short_name }))
			.sort((a, b) => b.p - a.p)
	);

	let map = $state<{ children: Children; profiles: Profiles } | null>(null);
	let failure = $state<string | null>(null);
	let colorBy = $state('group');

	const views = [
		{ value: 'group', label: 'Group' },
		{ value: 'adhd', label: 'ADHD' },
		{ value: 'speech', label: 'Speech or language disorder' }
	];

	onMount(async () => {
		try {
			const [children, profiles] = await Promise.all([loadData<Children>('children.json'), loadData<Profiles>('profiles.json')]);
			map = { children, profiles };
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
		}
	});
</script>

<svelte:head>
	<title>Subtypes or scales? Co-occurring diagnoses in autistic children</title>
</svelte:head>

<article>
	<section class="opening wide">
		<h1>{TITLE}</h1>
		<p class="lede prose">
			Nearly all autistic children ({percent(data.numbers.share_with_any_condition.value)} in this survey) have other diagnoses too.
			Studies that sort them by those diagnoses usually report <a href={LITMAN}>three or four subtypes</a>, and
			<a href={AGELINK}>few check</a> that the subtypes hold up. We found four groups among {n} children, and they held up in later survey
			years, in the 2025 survey and under resampling. Then we asked whether that proves they are real.
		</p>
	</section>

	<section class="wide step first" aria-labelledby="one-row">
		<h2 id="one-row">One row of answers</h2>
		<p class="prose">
			The National Survey of Children's Health asks parents which of 18 conditions a doctor or other provider has diagnosed in their
			child. This exact row of answers is shared by {example.count} children:
		</p>
		<div class="example">
			<ProfileRow answers={example.answers} conditions={data.model.conditions} />
		</div>
		<p class="prose">
			A model that sorts children into four groups cannot decide which group fits these answers. Its best fit is
			{first.name}, at {percent(first.p)}, with {second.name} close behind at {percent(second.p)}.
		</p>
		<div class="bar">
			<ProbabilityBar probabilities={example.class_probabilities} groups={data.model.classes} />
		</div>
		<p class="aside">
			For each condition the survey asks whether a provider has ever said the child has it and, for most, whether the child has it
			now; a diagnosis counts here when both answers are yes. Five lifelong conditions (cerebral palsy, blood disorders, Down
			syndrome, hearing problems and vision problems) have no "now" question, so the first answer counts. These are diagnoses a
			parent reports: a child with fewer of them may have been diagnosed less often, which depends on things like access to
			specialists as well as on the child.
		</p>
	</section>

	<section class="wide step" aria-labelledby="four-groups">
		<h2 id="four-groups">Four groups</h2>
		<p class="prose">
			Asked for four groups, a model splits the {n} children into four. Each dot below is one child, placed by two scores built from
			their answers. One score rises with developmental delay, learning disability, speech or language disorder and intellectual
			disability; the other with anxiety, depression, ADHD and behavior problems.
		</p>

		{#if failure}
			<p role="alert">The map could not load. {failure} Reloading the page usually fixes this.</p>
		{:else if !map}
			<p class="loading">Loading the {n} children.</p>
		{:else}
			<div class="controls" role="group" aria-labelledby="color-by">
				<span id="color-by">Color the children by</span>
				{#each views as view}
					<button type="button" aria-pressed={colorBy === view.value} onclick={() => (colorBy = view.value)}>{view.label}</button>
				{/each}
			</div>
			<p class="hint">Color by ADHD and the dark dots climb; color by speech or language disorder and they run to the right.</p>
			<ChildMap children={map.children} profiles={map.profiles} model={data.model} {colorBy} label="Map of {n} autistic children" />
			<p class="caption">
				Each child's position is an estimate drawn from their answers, so children with identical answers land near each other, not
				on one spot.
				{#if colorBy === 'group'}
					Faded dots are children whose best-fitting group has less than 80% probability. The percentages are each group's
					estimated share of US children aged 6 to 17 whose parent reports a current autism diagnosis.
				{/if}
				This view assumes positions follow a bell curve, which pulls any data into one round cloud, so the absence of gaps here is not
				evidence either way. We come back to that below.
			</p>
		{/if}

		<div class="rates">
			<p class="prose">
				The names describe the diagnoses that are common in each group. Each column is one group; the bars show how many of its
				children have each diagnosis.
			</p>
			<GroupRates model={data.model} />
		</div>
	</section>

	<section class="wide step todo" aria-labelledby="todo">
		<h2 id="todo">Mockup ends here</h2>
		<p>
			Still to build: the groups replicate; three clouds, one with real groups; children with no groups pass the same checks; three
			more checks; what this can and cannot say.
		</p>
	</section>
</article>

<style>
	.opening {
		padding-block: clamp(2rem, 5vw, 3.5rem) 0;
	}
	.opening h1 {
		max-width: 56rem;
	}
	.lede {
		font-size: var(--step-1);
		line-height: 1.55;
		max-width: 40rem;
	}
	.step {
		padding-block: 3rem 1rem;
	}
	.step.first {
		padding-top: 1.5rem;
	}
	.example {
		margin-block: 1.5rem;
		padding: 1.25rem 0;
		border-block: 1px solid var(--rule);
	}
	.bar {
		max-width: 30rem;
		margin-bottom: 1.5rem;
	}
	.aside,
	.caption,
	.loading,
	.hint {
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.caption {
		margin-top: 0.75rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.5rem;
		margin-block: 1.5rem 0.5rem;
		font-size: var(--step--1);
	}
	.controls button {
		font: inherit;
		color: var(--ink);
		background: #fff;
		border: 1px solid var(--rule);
		border-radius: 999px;
		padding: 0.25rem 0.8rem;
		cursor: pointer;
	}
	.controls button[aria-pressed='true'] {
		background: var(--ink);
		border-color: var(--ink);
		color: #fff;
	}
	.hint {
		margin-bottom: 1rem;
	}
	.rates {
		margin-top: 3rem;
	}
	.todo {
		color: var(--ink-2);
	}
</style>
