<!-- One set of 2,500 children drawn as plain dots, under the bell-curve draw or the estimated-distribution draw. -->
<script lang="ts">
	import type { Cloud } from '#lib/data.ts';
	import { INK } from '#lib/groups.ts';
	import { linear } from '#lib/scale.ts';

	let { cloud, sharp = false, label }: { cloud: Cloud; sharp?: boolean; label: string } = $props();

	const LIMIT = 4;
	let width = $state(0);
	let canvas: HTMLCanvasElement | undefined = $state();

	$effect(() => {
		if (!canvas || width === 0) return;
		const size = width;
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		canvas.width = Math.round(size * dpr);
		canvas.height = Math.round(size * dpr);
		const ctx = canvas.getContext('2d')!;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, size, size);
		const sx = linear([-LIMIT, LIMIT], [0, size]);
		const sy = linear([-LIMIT, LIMIT], [size, 0]);
		const xs = sharp ? cloud.sharp_x : cloud.x;
		const ys = sharp ? cloud.sharp_y : cloud.y;
		const r = Math.max(1, size / 220);
		ctx.fillStyle = INK;
		ctx.globalAlpha = 0.3;
		for (let i = 0; i < xs.length; i++) {
			ctx.beginPath();
			ctx.arc(sx(xs[i]), sy(ys[i]), r, 0, 2 * Math.PI);
			ctx.fill();
		}
	});
</script>

<div class="cloud" bind:clientWidth={width} role="img" aria-label={label}>
	<canvas bind:this={canvas} style:width="{width}px" style:height="{width}px" aria-hidden="true"></canvas>
</div>

<style>
	.cloud {
		aspect-ratio: 1;
		background: #fff;
		outline: 1px solid var(--rule);
		border-radius: 4px;
	}
	canvas {
		display: block;
	}
</style>
