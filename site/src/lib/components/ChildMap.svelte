<!--
	Every child as a dot on the two scales, drawn on a canvas, colored by best-fitting group
	or by one diagnosis. Pointing at a dot shows a short description. A click or tap (or,
	with the map focused, the arrow keys and Enter) pins it, so the full list of diagnoses
	can be opened. Pins happen on click, not pointer-down, so a swipe that starts on the map
	scrolls the page without pinning anything.
-->
<script lang="ts">
	import { quadtree } from 'd3-quadtree';
	import type { Children, Model, Profiles } from '#lib/data.ts';
	import { abbreviate, describeProfile, diagnosesText, mostLikely, sharedText, summarize } from '#lib/describe.ts';
	import { CONDITION_SHORT, CONFIDENT, FAINT, GROUP_COLORS, INK, inSentence, percent } from '#lib/groups.ts';
	import { linear } from '#lib/scale.ts';

	interface Props {
		children: Children;
		profiles: Profiles;
		model: Model;
		/** "group", or the flag of one diagnosis. */
		colorBy?: string;
		label: string;
	}
	let { children, profiles, model, colorBy = 'group', label }: Props = $props();

	const id = $props.id();
	const LIMIT = 4;
	const STEP = 0.2;
	const MAX = 620;

	let width = $state(0);
	let plot: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let overlay: HTMLCanvasElement | undefined = $state();
	let hovered: number | null = $state(null);
	let pinned: number | null = $state(null);
	let expanded = $state(false);
	let cursor: [number, number] | null = $state(null);
	/** What screen readers hear; set only by keyboard moves and pins, never by the mouse. */
	let readout = $state('');

	const size = $derived(Math.max(240, Math.min(width, MAX)));
	const sx = $derived(linear([-LIMIT, LIMIT], [0, size]));
	const sy = $derived(linear([-LIMIT, LIMIT], [size, 0]));
	const radius = $derived(Math.max(1.3, size / 310));
	/** On narrow screens group labels sit in a legend and the description sits under the map. */
	const compact = $derived(size < 440);
	const conditionIndex = $derived(model.conditions.findIndex((c) => c.flag === colorBy));
	const conditionName = $derived(conditionIndex < 0 ? '' : inSentence(CONDITION_SHORT[colorBy]));

	/** Fill, opacity and drawing order for every child. Dots that matter are drawn last. */
	const style = $derived.by(() => {
		const n = children.n;
		const fill: string[] = new Array(n);
		const alpha = new Float32Array(n);
		const emphasis = new Uint8Array(n);
		for (let i = 0; i < n; i++) {
			const p = children.profile[i];
			if (conditionIndex < 0) {
				const { group, probability } = mostLikely(p, profiles);
				const sure = probability >= CONFIDENT;
				fill[i] = GROUP_COLORS[group];
				alpha[i] = sure ? 0.8 : 0.28;
				emphasis[i] = sure ? 1 : 0;
			} else {
				const yes = profiles.answers[p][conditionIndex] === '1';
				fill[i] = yes ? INK : FAINT;
				alpha[i] = yes ? 0.75 : 0.6;
				emphasis[i] = yes ? 1 : 0;
			}
		}
		const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => emphasis[a] - emphasis[b]);
		return { fill, alpha, order };
	});

	const tree = $derived(
		quadtree<number>()
			.x((i) => sx(children.x[i]))
			.y((i) => sy(children.y[i]))
			.addAll(Array.from({ length: children.n }, (_, i) => i))
	);

	/** The dot being described: the pinned one, otherwise the one under the pointer. */
	const shown = $derived(pinned ?? hovered);
	const description = $derived(shown === null ? null : describeProfile(children.profile[shown], profiles, model));
	const more = $derived(description ? abbreviate(description.diagnoses).more : 0);

	const corners = $derived(
		model.classes.map((g, i) => ({ ...g, color: GROUP_COLORS[i], right: g.center[0] > 0, top: g.center[1] > 0 }))
	);

	function prepare(c: HTMLCanvasElement) {
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		c.width = Math.round(size * dpr);
		c.height = Math.round(size * dpr);
		const ctx = c.getContext('2d')!;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, size, size);
		return ctx;
	}

	$effect(() => {
		if (!canvas || width === 0) return;
		const ctx = prepare(canvas);
		const { fill, alpha, order } = style;
		for (const i of order) {
			ctx.globalAlpha = alpha[i];
			ctx.fillStyle = fill[i];
			ctx.beginPath();
			ctx.arc(sx(children.x[i]), sy(children.y[i]), radius, 0, 2 * Math.PI);
			ctx.fill();
		}
		ctx.globalAlpha = 1;
	});

	$effect(() => {
		if (!overlay || width === 0) return;
		const ctx = prepare(overlay);
		if (cursor) {
			ctx.strokeStyle = INK;
			ctx.setLineDash([3, 3]);
			ctx.beginPath();
			ctx.arc(sx(cursor[0]), sy(cursor[1]), 14, 0, 2 * Math.PI);
			ctx.stroke();
			ctx.setLineDash([]);
		}
		if (shown === null) return;
		const x = sx(children.x[shown]);
		const y = sy(children.y[shown]);
		ctx.lineWidth = 2;
		ctx.strokeStyle = '#ffffff';
		ctx.beginPath();
		ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.lineWidth = 1.5;
		ctx.strokeStyle = INK;
		ctx.beginPath();
		ctx.arc(x, y, radius + 3, 0, 2 * Math.PI);
		ctx.stroke();
	});

	function announce(index: number | null) {
		readout = index === null ? '' : summarize(describeProfile(children.profile[index], profiles, model), children.n);
	}

	function nearest(event: MouseEvent): number | null {
		const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
		return tree.find(event.clientX - rect.left, event.clientY - rect.top, 12) ?? null;
	}

	function move(event: PointerEvent) {
		if (pinned !== null || event.pointerType === 'touch') return;
		hovered = nearest(event);
		cursor = null;
	}

	function click(event: MouseEvent) {
		if ((event.target as HTMLElement).closest('.tip')) return;
		pin(nearest(event));
	}

	function pin(index: number | null) {
		pinned = index;
		hovered = index;
		expanded = false;
		announce(index);
	}

	function unpin() {
		pin(null);
		cursor = null;
		plot?.focus();
	}

	function key(event: KeyboardEvent) {
		// Keys pressed on the description's buttons belong to the buttons, except Escape.
		if (event.target !== event.currentTarget && event.key !== 'Escape') return;
		if (event.key === 'Escape') {
			unpin();
			return;
		}
		if (event.key === 'Enter') {
			pin(pinned === null ? hovered : null);
			return;
		}
		const moves: Record<string, [number, number]> = { ArrowLeft: [-STEP, 0], ArrowRight: [STEP, 0], ArrowUp: [0, STEP], ArrowDown: [0, -STEP] };
		const step = moves[event.key];
		if (!step) return;
		event.preventDefault();
		const [cx, cy] = cursor ?? [0, 0];
		cursor = [Math.max(-LIMIT, Math.min(LIMIT, cx + step[0])), Math.max(-LIMIT, Math.min(LIMIT, cy + step[1]))];
		pinned = null;
		expanded = false;
		hovered = tree.find(sx(cursor[0]), sy(cursor[1]), 24) ?? null;
		announce(hovered);
	}

	const tipLeft = $derived(shown === null ? 0 : sx(children.x[shown]));
	const tipTop = $derived(shown === null ? 0 : sy(children.y[shown]));
</script>

{#snippet details()}
	{#if description}
		<p class="tip-head">{diagnosesText(description.diagnoses, expanded)}</p>
		<p class="tip-note">{sharedText(description.sharedBy, children.n)}</p>
		<p class="tip-note">Four-group model's best fit for these answers:</p>
		<ul>
			{#each description.chances.filter((c) => c.probability >= 0.05) as c}
				<li>
					<span class="key" style:background={GROUP_COLORS[c.group]}></span>{c.name}
					<span class="num">{percent(c.probability)}</span>
				</li>
			{/each}
		</ul>
		{#if pinned === null}
			{#if more > 0}<p class="tip-hint">Click, tap or press Enter to see all diagnoses.</p>{/if}
		{:else}
			<div class="tip-actions">
				{#if more > 0}
					<button type="button" onclick={() => (expanded = !expanded)}>{expanded ? 'Show fewer' : 'Show all'}</button>
				{/if}
				<button type="button" onclick={unpin}>Close</button>
			</div>
		{/if}
	{/if}
{/snippet}

<figure class="map">
	<div class="frame" class:compact>
		{#if compact}
			<p class="y-top">More mental-health and behavioral diagnoses, toward the top</p>
		{:else}
			<p class="y-label" style:height="{size}px"><span>Fewer</span><span>More mental-health and behavioral diagnoses</span></p>
		{/if}
		<div class="column" bind:clientWidth={width}>
			<!-- The map is a canvas that answers the pointer and the arrow keys; no native element
			     does that, so it is an "application" region with its own keyboard handling. -->
			<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions, a11y_click_events_have_key_events -->
			<div
				class="plot"
				bind:this={plot}
				style:width="{size}px"
				style:height="{size}px"
				role="application"
				aria-label="{label}. Use the arrow keys to move between children, Enter to pin one, Escape to clear."
				aria-describedby="{id}-readout"
				tabindex="0"
				onpointermove={move}
				onclick={click}
				onpointerleave={() => {
					if (pinned === null) hovered = null;
				}}
				onkeydown={key}
				onblur={(event) => {
					if ((event.relatedTarget as HTMLElement | null)?.closest('.tip')) return;
					cursor = null;
					if (pinned === null) hovered = null;
				}}
			>
				<canvas bind:this={canvas} style:width="{size}px" style:height="{size}px"></canvas>
				<canvas class="overlay" bind:this={overlay} style:width="{size}px" style:height="{size}px"></canvas>

				{#if conditionIndex < 0 && !compact}
					{#each corners as g}
						<p class="corner" class:right={g.right} class:top={g.top}>
							<span class="key" style:background={g.color}></span><span class="name">{g.short_name}</span>
							<span class="share num">{percent(g.weighted_share)}</span>
						</p>
					{/each}
				{/if}

				{#if description && !compact}
					<div
						class="tip"
						class:pinned={pinned !== null}
						class:flip={tipLeft > size * 0.55}
						class:below={tipTop < size * 0.3}
						style:left="{tipLeft}px"
						style:top="{tipTop}px"
					>
						{@render details()}
					</div>
				{/if}
			</div>
			<p class="x-label" style:width="{size}px">
				<span>Fewer</span><span class="more">More developmental and language diagnoses</span>
			</p>
			{#if description && compact}
				<div class="tip static" class:pinned={pinned !== null}>{@render details()}</div>
			{/if}
		</div>
	</div>
	<p id="{id}-readout" class="visually-hidden" aria-live="polite">{readout}</p>
	{#if conditionIndex < 0 && compact}
		<ul class="legend">
			{#each corners as g}
				<li><span class="key" style:background={g.color}></span>{g.short_name}&nbsp;<span class="share num">{percent(g.weighted_share)}</span></li>
			{/each}
		</ul>
	{:else if conditionIndex >= 0}
		<ul class="legend">
			<li><span class="key" style:background={INK}></span>{conditionName.charAt(0).toUpperCase() + conditionName.slice(1)} reported</li>
			<li><span class="key" style:background={FAINT}></span>No {conditionName} reported</li>
		</ul>
	{/if}
</figure>

<style>
	.map {
		margin: 0;
	}
	.frame {
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: start;
		column-gap: 0.5rem;
		max-width: calc(620px + 2rem);
	}
	.frame.compact {
		grid-template-columns: 1fr;
	}
	.column {
		min-width: 0;
	}
	.y-top {
		margin: 0 0 0.4rem;
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.y-label {
		display: flex;
		justify-content: space-between;
		writing-mode: vertical-rl;
		transform: rotate(180deg);
		margin: 0;
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.x-label {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		margin: 0.4rem 0 0;
		max-width: none;
		font-size: var(--step--1);
		color: var(--ink-2);
	}
	.x-label .more {
		text-align: right;
	}
	.plot {
		position: relative;
		box-sizing: content-box;
		touch-action: pan-y;
		background: #fff;
		outline: 1px solid var(--rule);
		border-radius: 4px;
		cursor: crosshair;
	}
	.plot:focus-visible {
		outline: 2px solid var(--ink);
		outline-offset: 3px;
	}
	canvas {
		position: absolute;
		inset: 0;
		display: block;
	}
	.overlay {
		pointer-events: none;
	}
	.corner {
		position: absolute;
		left: 0.6rem;
		bottom: 0.5rem;
		margin: 0;
		font-size: var(--step--1);
		line-height: 1.3;
		background: rgb(255 255 255 / 0.85);
		padding: 0.1rem 0.35rem;
		border-radius: 3px;
		pointer-events: none;
	}
	.corner.right {
		left: auto;
		right: 0.6rem;
		text-align: right;
	}
	.corner.top {
		bottom: auto;
		top: 0.5rem;
	}
	.name {
		font-weight: 600;
	}
	.share {
		color: var(--ink-2);
		margin-left: 0.3rem;
	}
	.key {
		display: inline-block;
		width: 0.65rem;
		height: 0.65rem;
		border-radius: 50%;
		margin-right: 0.4rem;
	}
	.tip {
		position: absolute;
		z-index: 2;
		width: 16rem;
		transform: translate(14px, -50%);
		background: #fff;
		border: 1px solid var(--rule);
		border-radius: 4px;
		box-shadow: 0 2px 8px rgb(28 34 48 / 0.12);
		padding: 0.6rem 0.75rem;
		font-size: var(--step--1);
		line-height: 1.4;
		pointer-events: none;
	}
	.tip.flip {
		transform: translate(calc(-100% - 14px), -50%);
	}
	.tip.below {
		transform: translate(14px, 0);
	}
	.tip.flip.below {
		transform: translate(calc(-100% - 14px), 0);
	}
	.tip.pinned {
		pointer-events: auto;
	}
	.tip.static {
		position: static;
		width: auto;
		transform: none;
		margin-top: 0.75rem;
		box-shadow: none;
	}
	.tip p {
		margin: 0 0 0.35rem;
	}
	.tip-head {
		font-weight: 600;
	}
	.tip-note {
		color: var(--ink-2);
	}
	.tip-hint {
		color: var(--ink-2);
		margin: 0.4rem 0 0 !important;
	}
	.tip ul,
	.legend {
		list-style: none;
		margin: 0;
		padding: 0;
	}
	.tip li {
		display: flex;
		align-items: center;
	}
	.tip li .num {
		margin-left: auto;
		padding-left: 1rem;
	}
	.tip-actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.5rem;
	}
	.tip-actions button {
		font: inherit;
		color: var(--ink);
		background: var(--bg);
		border: 1px solid var(--rule);
		border-radius: 4px;
		padding: 0.15rem 0.6rem;
		cursor: pointer;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem 1.25rem;
		margin-top: 0.75rem;
		font-size: var(--step--1);
	}
</style>
