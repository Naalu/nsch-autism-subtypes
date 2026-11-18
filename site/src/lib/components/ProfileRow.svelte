<!--
	One set of 18 answers, shown as three families of yes/no marks. Used for combinations
	shared by enough children that their answers are published.
-->
<script lang="ts">
	import type { Condition } from '#lib/data.ts';
	import { CONDITION_FAMILIES, CONDITION_SHORT } from '#lib/groups.ts';

	let { answers, conditions }: { answers: string; conditions: Condition[] } = $props();

	const has = $derived(Object.fromEntries(conditions.map((c, j) => [c.flag, answers[j] === '1'])));
</script>

<div class="row" role="list" aria-label="The 18 conditions the survey asks about">
	{#each CONDITION_FAMILIES as family}
		<div class="family" role="listitem">
			<p class="family-name">{family.name}</p>
			<ul>
				{#each family.flags as flag}
					<li class:yes={has[flag]}>
						<span class="mark" aria-hidden="true"></span>
						<span class="label">{CONDITION_SHORT[flag]}</span>
						<span class="visually-hidden">{has[flag] ? 'yes' : 'no'}</span>
					</li>
				{/each}
			</ul>
		</div>
	{/each}
</div>

<style>
	.row {
		display: grid;
		grid-template-columns: 1fr 1fr 2.4fr;
		gap: 1.25rem 2rem;
	}
	@media (max-width: 52rem) {
		.row {
			grid-template-columns: 1fr 1fr;
		}
		.family:last-child {
			grid-column: 1 / -1;
		}
	}
	.family-name {
		font-size: var(--step--1);
		color: var(--ink-2);
		margin: 0 0 0.5rem;
	}
	ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 0.3rem 1.25rem;
	}
	.family:last-child ul {
		grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
	}
	li {
		display: flex;
		align-items: center;
		gap: 0.55rem;
		color: var(--ink-2);
		font-size: var(--step--1);
	}
	.mark {
		flex: none;
		width: 0.95rem;
		height: 0.95rem;
		border: 1.5px solid var(--faint);
		border-radius: 3px;
	}
	li.yes {
		color: var(--ink);
		font-weight: 600;
	}
	li.yes .mark {
		background: var(--ink);
		border-color: var(--ink);
	}
</style>
