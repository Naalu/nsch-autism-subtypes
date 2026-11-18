<script lang="ts">
	import '@fontsource/ibm-plex-sans/400.css';
	import '@fontsource/ibm-plex-sans/600.css';
	import '@fontsource/ibm-plex-sans/700.css';
	import '@fontsource/ibm-plex-serif/400.css';
	import '@fontsource/ibm-plex-serif/400-italic.css';
	import '@fontsource/ibm-plex-serif/600.css';
	import '../app.css';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';

	let { children } = $props();

	const pages = [
		{ route: '/', href: resolve('/'), label: 'Story' },
		{ route: '/explore', href: resolve('explore/'), label: 'Explore' },
		{ route: '/methods', href: resolve('methods/'), label: 'Methods and paper' }
	] as const;
</script>

<header class="wide">
	<a class="title" href={resolve('/')}>Co-occurring diagnoses in autistic children</a>
	<nav aria-label="Pages">
		{#each pages as p}
			<a href={p.href} aria-current={page.route.id === p.route ? 'page' : undefined}>{p.label}</a>
		{/each}
	</nav>
	<p class="notice">
		A class project, not peer reviewed. Each dot is one child's survey answers, with no year, age or place attached. Nothing here can
		tell you which group, if any, a particular child is in.
	</p>
</header>

<main>
	{@render children()}
</main>

<footer class="wide">
	<p>
		A class project for CS 472/572 (Unsupervised Learning) at Northern Arizona University, by Chris Reger and Haley Berger. Not peer
		reviewed. The figures describe parents' survey answers; they cannot tell you which group, if any, a particular child is in.
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
