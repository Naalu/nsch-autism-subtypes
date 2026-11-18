<script lang="ts">
	import { onMount } from 'svelte';
	import ChildMap from '#lib/components/ChildMap.svelte';
	import ProbabilityBar from '#lib/components/ProbabilityBar.svelte';
	import ProfileRow from '#lib/components/ProfileRow.svelte';
	import { loadData, type Children, type Model, type Numbers, type Profiles } from '#lib/data.ts';
	import { CONDITION_SHORT, CONFIDENT, percent } from '#lib/groups.ts';

	// The opening example: a common combination the four-group model nearly splits in half.
	const EXAMPLE = '110000000000000000';

	type SiteData = { children: Children; profiles: Profiles; model: Model; numbers: Numbers };
	let data = $state<SiteData | null>(null);
	let failure = $state<string | null>(null);
	let colorBy = $state('group');

	onMount(async () => {
		try {
			const [children, profiles, model, numbers] = await Promise.all([
				loadData<Children>('children.json'),
				loadData<Profiles>('profiles.json'),
				loadData<Model>('model.json'),
				loadData<Numbers>('numbers.json')
			]);
			data = { children, profiles, model, numbers };
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
		}
	});

	const example = $derived(data ? data.profiles.answers.indexOf(EXAMPLE) : -1);
	/** Share of children whose most likely group has less than 80% probability. */
	const uncertain = $derived.by(() => {
		if (!data) return 0.37;
		const { children, profiles } = data;
		let count = 0;
		for (const p of children.profile) if (Math.max(...profiles.class_probabilities[p]) < CONFIDENT) count++;
		return count / children.n;
	});
	const n = $derived(data ? data.numbers.children.value.toLocaleString('en-US') : '8,891');
</script>

<svelte:head>
	<title>Autistic children's co-occurring diagnoses: types or two scales?</title>
</svelte:head>

<article>
	<section class="opening wide">
		<h1>Do autistic children's other diagnoses sort them into types?</h1>
		<p class="lede">
			Most autistic children have other diagnoses too: ADHD, anxiety, a learning disability, a speech disorder. Studies that sort
			children by these diagnoses usually report three or four groups, and few check whether the groups come back in a new set of
			children. We sorted {n} children from a national survey of parents. Four groups came out, and they came back every time we
			looked again. Then we asked whether that proves the groups are real.
		</p>
	</section>

	<section class="wide step" aria-labelledby="one-row">
		<h2 id="one-row">One row of answers</h2>
		<p>
			The survey asks parents whether a doctor or other health care provider has ever told them their child has each of 18 conditions,
			and if so, whether the child has it now. A condition counts here when the answer to both is yes. This exact row of answers is
			shared by {data && example >= 0 ? data.profiles.count[example] : 147} children:
		</p>
		{#if data && example >= 0}
			<div class="example">
				<ProfileRow answers={EXAMPLE} conditions={data.model.conditions} />
			</div>
			<p>
				A model that sorts children into four groups cannot decide where these children belong. It gives them a
				{percent(data.profiles.class_probabilities[example][2])} chance of the mental-health and behavior group and
				{percent(data.profiles.class_probabilities[example][0])} of the group with fewer diagnoses.
			</p>
			<div class="bar">
				<ProbabilityBar probabilities={data.profiles.class_probabilities[example]} groups={data.model.classes} />
			</div>
			<p>
				If children came in types, most rows of answers would point clearly to one group. This one does not, and it is not unusual:
				for {percent(uncertain)} of the {n} children, no group gets as much as 80%.
			</p>
		{/if}
		<p class="aside">
			These are diagnoses a parent reports. A child with fewer of them may have been diagnosed less often, which depends on things
			like access to specialists as well as on the child.
		</p>
	</section>

	<section class="wide step" aria-labelledby="four-groups">
		<h2 id="four-groups">Four groups on two scales</h2>
		<p>
			Sorted all at once, the {n} children form four groups. The groups line up with two scales that a second model finds in the same
			answers. One rises with developmental delay, learning disability, speech or language disorder and intellectual disability. The
			other rises with anxiety, depression, ADHD and behavior problems. Each dot is one child, placed where their answers put them on
			the two scales.
		</p>

		{#if failure}
			<p role="alert">The map could not load. {failure} Reloading the page usually fixes this.</p>
		{:else if !data}
			<p class="loading">Loading the {n} children.</p>
		{:else}
			<div class="controls">
				<label for="color-by">Color the children by</label>
				<select id="color-by" bind:value={colorBy}>
					<option value="group">their most likely group</option>
					<optgroup label="one condition">
						{#each data.model.conditions as c}
							<option value={c.flag}>{CONDITION_SHORT[c.flag]}</option>
						{/each}
					</optgroup>
				</select>
			</div>
			<ChildMap
				children={data.children}
				profiles={data.profiles}
				model={data.model}
				{colorBy}
				label="Map of {n} autistic children on the two scales"
			/>
			<p class="caption">
				Each child's position is an estimate drawn from their answers, so it is approximate.
				{#if colorBy === 'group'}
					Faded dots are children the model places in a group with less than 80% probability. The percentages are each group's
					estimated share of US autistic children aged 6 to 17.
				{/if}
				This view assumes positions follow a bell curve, which pulls any data into one round cloud, so the absence of gaps here is not
				evidence either way. Step 4 comes back to that.
			</p>
		{/if}
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
		padding-block: clamp(3rem, 8vw, 6rem) 2rem;
	}
	.opening h1 {
		max-width: 20ch;
	}
	.lede {
		font-size: var(--step-1);
		line-height: 1.55;
		max-width: 40rem;
	}
	.step {
		padding-block: 3rem 1rem;
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
	.loading {
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.caption {
		margin-top: 0.75rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.5rem;
		margin-block: 1.5rem 1rem;
		font-size: var(--step--1);
	}
	select {
		font: inherit;
		color: var(--ink);
		background: #fff;
		border: 1px solid var(--rule);
		border-radius: 4px;
		padding: 0.3rem 0.5rem;
	}
	.todo {
		color: var(--ink-2);
	}
</style>
