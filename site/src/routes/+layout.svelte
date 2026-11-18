<script lang="ts">
	import '@fontsource/atkinson-hyperlegible-next/400.css';
	import '@fontsource/atkinson-hyperlegible-next/600.css';
	import '@fontsource/atkinson-hyperlegible-next/700.css';
	import '../app.css';
	import { base } from '$app/paths';
	import { page } from '$app/state';

	let { children } = $props();

	const pages = [
		{ href: `${base}/`, label: 'Story' },
		{ href: `${base}/explore/`, label: 'Explore' },
		{ href: `${base}/methods/`, label: 'Methods and paper' }
	];
</script>

<header class="wide">
	<a class="title" href="{base}/">Types or degrees?</a>
	<nav aria-label="Pages">
		{#each pages as p}
			<a href={p.href} aria-current={page.url.pathname === p.href ? 'page' : undefined}>{p.label}</a>
		{/each}
	</nav>
	<p class="notice">A class project, not peer reviewed. Nothing here describes or diagnoses an individual child.</p>
</header>

<main>
	{@render children()}
</main>

<footer class="wide">
	<p>
		A class project for CS 472/572 (Unsupervised Learning) at Northern Arizona University, by Chris Reger and Haley Berger. Not peer
		reviewed. The figures describe survey answers about groups of children and say nothing about any individual child.
	</p>
	<p>
		Data: National Survey of Children's Health, 2016 to 2024, public-use files.
		<a href="https://github.com/Naalu/nsch-autism-subtypes">Code, paper and every number on this site</a>.
	</p>
</footer>

<style>
	header {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.5rem 1.5rem;
		padding-block: 1.25rem;
		border-bottom: 1px solid var(--rule);
	}
	.title {
		font-weight: 700;
		text-decoration: none;
	}
	nav {
		display: flex;
		gap: 1.25rem;
		font-size: var(--step--1);
	}
	nav a {
		text-decoration: none;
		color: var(--ink-2);
		padding-block: 0.25rem;
	}
	nav a[aria-current='page'] {
		color: var(--ink);
		box-shadow: inset 0 -2px 0 var(--ink);
	}
	.notice {
		flex-basis: 100%;
		max-width: none;
		margin: 0;
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	footer {
		margin-top: 6rem;
		padding-block: 2rem 3rem;
		border-top: 1px solid var(--rule);
		font-size: var(--step--1);
		color: var(--ink-2);
	}
</style>
