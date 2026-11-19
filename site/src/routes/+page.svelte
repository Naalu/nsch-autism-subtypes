<script lang="ts">
	import { onMount } from 'svelte';
	import ChildMap from '#lib/components/ChildMap.svelte';
	import CloudGuess from '#lib/components/CloudGuess.svelte';
	import ReplicationStrip from '#lib/components/ReplicationStrip.svelte';
	import YearMaps from '#lib/components/YearMaps.svelte';
	import GroupRates from '#lib/components/GroupRates.svelte';
	import ProbabilityBar from '#lib/components/ProbabilityBar.svelte';
	import ProfileRow from '#lib/components/ProfileRow.svelte';
	import { loadData, type Children, type Clouds, type Profiles } from '#lib/data.ts';
	import { againstRange, checkRows, position, shapeRanges } from '#lib/evidence.ts';
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

	const rows = $derived(checkRows(data.evidence.replication));
	const shape = $derived(shapeRanges(data.evidence.shape));
	const k4 = $derived(data.evidence.checks_by_k.find((row) => row.K === 4)!);
	/** Checks where the real children fall inside the no-groups range, and where they fall outside it. */
	const inside = $derived(rows.filter((r) => position(r.real, r.noGroups) === 'within'));
	const outside = $derived(rows.filter((r) => position(r.real, r.noGroups) !== 'within'));
	const belowFour = $derived(rows.filter((r) => position(r.real, r.fourGroups) === 'below'));
	const two = (v: number) => v.toFixed(2);
	const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);

	let map = $state<{ children: Children; profiles: Profiles } | null>(null);
	let clouds = $state<Clouds | null>(null);
	let failure = $state<string | null>(null);
	let colorBy = $state('group');

	const views = [
		{ value: 'group', label: 'Group' },
		{ value: 'adhd', label: 'ADHD' },
		{ value: 'speech', label: 'Speech or language disorder' }
	];

	onMount(async () => {
		try {
			const [children, profiles, cloudData] = await Promise.all([
				loadData<Children>('children.json'),
				loadData<Profiles>('profiles.json'),
				loadData<Clouds>('clouds.json')
			]);
			map = { children, profiles };
			clouds = cloudData;
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

	<section class="wide step" aria-labelledby="hold-up">
		<h2 id="hold-up">They hold up</h2>
		<p class="prose">
			A grouping that turns up in one sample and nowhere else is noise, so we checked whether these four come back. Each check sorts
			the children a second time and measures agreement between the two sortings, from 0 (no better than chance) to 1 (identical).
		</p>
		<p class="prose">
			Sorted using only some survey years, the children of the other years got the same groups with agreement
			{two(k4['held-out years'])}. Sorted on 2016 to 2024 and applied to the 2025 survey, which we held back until the end:
			{two(k4['2025 forward test'])}. And when we resampled the children over and over, even the least stable group kept an overlap
			of {two(k4['least stable class (Jaccard)'])} with itself.
		</p>
		<YearMaps years={data.years} model={data.model} />
		<p class="caption">
			Left: each square is a patch of the map for one survey year, colored by the most common group there; stronger squares hold
			more children, and patches with fewer than {data.years.min_cell} children are left out. Right: each group's estimated share
			of US children aged 6 to 17 whose parent reports a current autism diagnosis, with 95% intervals.
		</p>
		<p class="prose">
			By the usual standard, this is where a subtype study ends: four groups in a national sample that come back year after year
			and in a new survey.
		</p>
	</section>

	<section class="wide step" aria-labelledby="three-clouds">
		<h2 id="three-clouds">Three clouds</h2>
		<p class="prose">
			Here is a harder test. Below are three sets of 2,500 children, drawn the way the map above draws them. One is a sample of the
			real survey. One is made up from four separate groups with clear gaps between them. One is made up with no groups at all:
			every made-up child sits somewhere on the same two smooth scales.
		</p>
		{#if clouds}
			<CloudGuess {clouds} />
		{:else if !failure}
			<p class="loading">Loading the three sets of children.</p>
		{/if}
		<p class="prose after-guess">
			The usual drawing assumes the scores follow a bell curve, and that assumption pulls any data into one round cloud. That is
			why the map above cannot show gaps. Measured directly, by how much better a lumpy distribution fits each set than a bell
			curve does, the made-up children with no groups score {Math.round(shape.continuum.min)} to {Math.round(shape.continuum.max)},
			the made-up children with four separate groups {Math.round(shape.classes.min)} to {Math.round(shape.classes.max)}, and the real
			children {Math.round(shape.real)}. The real children are lumpier than a smooth continuum and much smoother than four separate
			groups.
		</p>
	</section>

	<section class="wide step" aria-labelledby="turn">
		<h2 id="turn">Children with no groups pass too</h2>
		<p class="prose">
			Now the checks from above, run on made-up children: five sets with no groups and five with four separate groups, each the
			same size as the survey. If the four groups were real, the real children should score like the made-up four groups.
		</p>
		<ReplicationStrip {rows} />
		<p class="prose">
			They do not. The real children score below the made-up four groups on {belowFour.length === rows.length
				? 'every check'
				: `${belowFour.length} of ${rows.length} checks`}. Made-up children with no groups pass about as well as the real ones: their
			range takes in the real score on {inside.length} of the {rows.length} checks{#if outside.length},
				and the exceptions point both ways ({outside
					.map((r) => `${lower(r.label)}: ${againstRange(r.real, r.noGroups)}`)
					.join('; ')}){/if}.
		</p>
		<p class="prose">
			So passing these checks does not show that the groups are real. Children with no groups at all pass them too, and the real
			children never reach the scores that four separate groups produce.
		</p>
	</section>

	<section class="wide step todo" aria-labelledby="todo">
		<h2 id="todo">Still to build</h2>
		<p>Three more checks (how many groups, groups on top of the scales, outcomes) and what this can and cannot say.</p>
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
	.after-guess {
		margin-top: 1.5rem;
	}
</style>
