/**
 * Small pure helpers that turn the evidence files into what the story's figures draw.
 */
import type { CheckResult, Evidence, Years } from './data.ts';

/** The replication checks the story shows, in order, with plain-language labels. */
export const CHECKS = [
	{ key: 'held-out years', label: 'Fit on some survey years, compared on the others' },
	{ key: '2025 forward test', label: 'Fit on 2016 to 2024, compared on the 2025 survey' },
	{ key: 'age 6-11 refit', label: 'Fit again on ages 6 to 11 only' },
	{ key: 'age 12-17 refit', label: 'Fit again on ages 12 to 17 only' },
	{ key: 'second method (k-means)', label: 'A different method (k-means) on the same children' }
] as const;

export interface CheckRow {
	key: string;
	label: string;
	real: number;
	noGroups: number[];
	fourGroups: number[];
}

/** One row per check: the real agreement and each simulated replicate's, sorted. */
export function checkRows(replication: Evidence['replication']): CheckRow[] {
	const of = (data: CheckResult['data'], key: string) =>
		replication.simulated
			.filter((r) => r.data === data && r.check === key)
			.map((r) => r.ari)
			.sort((a, b) => a - b);
	return CHECKS.map(({ key, label }) => {
		const real = replication.real.find((r) => r.check === key);
		if (!real) throw new Error(`evidence.json has no real result for "${key}"`);
		return { key, label, real: real.ari, noGroups: of('continuum', key), fourGroups: of('4 classes', key) };
	});
}

/**
 * A value and a range in words, with enough decimals that a value outside the range never
 * reads as equal to its end ("0.541 against 0.442 to 0.540", not "0.54 against 0.44 to 0.54").
 */
export function againstRange(value: number, simulated: number[]): string {
	const low = Math.min(...simulated);
	const high = Math.max(...simulated);
	let digits = 2;
	while (digits < 4 && (value.toFixed(digits) === low.toFixed(digits) || value.toFixed(digits) === high.toFixed(digits))) digits++;
	return `${value.toFixed(digits)} against ${low.toFixed(digits)} to ${high.toFixed(digits)}`;
}

/**
 * Where the real value falls against a set of simulated values. A value within `tolerance`
 * of an end counts as inside: agreement differences that small are noise, and calling
 * 0.5411 "above" 0.5408 would overstate them.
 */
export function position(value: number, simulated: number[], tolerance = 0.005): 'below' | 'within' | 'above' {
	if (value < Math.min(...simulated) - tolerance) return 'below';
	if (value > Math.max(...simulated) + tolerance) return 'above';
	return 'within';
}

/** The most common group in a map cell, and its share of the cell. */
export function dominant(counts: number[]): { group: number; share: number; total: number } {
	let group = 0;
	let total = 0;
	for (let g = 0; g < counts.length; g++) {
		total += counts[g];
		if (counts[g] > counts[group]) group = g;
	}
	return { group, share: total ? counts[group] / total : 0, total };
}

/** Weighted shares by year as one series per group, in site order. */
export function sharesByGroup(years: Years, groupNames: string[]) {
	return groupNames.map((name) =>
		years.weighted_shares
			.filter((row) => row.class === name)
			.sort((a, b) => a.year - b.year)
			.map((row) => ({ year: row.year, share: row.share, low: row.ci_low, high: row.ci_high }))
	);
}

/** Smallest and largest of the shape check's improvement over a bell curve, per dataset. */
export function shapeRanges(shape: Evidence['shape']) {
	const range = (data: string) => {
		const values = shape.filter((s) => s.data === data).map((s) => s.improvement);
		return { min: Math.min(...values), max: Math.max(...values) };
	};
	return { real: range('real').min, continuum: range('continuum'), classes: range('classes') };
}
