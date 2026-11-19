import { describe, expect, it } from 'vitest';
import type { Evidence, Years } from './data.ts';
import { againstRange, checkRows, dominant, position, shapeRanges, sharesByGroup } from './evidence.ts';

const replication: Evidence['replication'] = {
	real: [
		{ data: 'real', replicate: 0, check: 'held-out years', ari: 0.71 },
		{ data: 'real', replicate: 0, check: '2025 forward test', ari: 0.79 },
		{ data: 'real', replicate: 0, check: 'age 6-11 refit', ari: 0.71 },
		{ data: 'real', replicate: 0, check: 'age 12-17 refit', ari: 0.77 },
		{ data: 'real', replicate: 0, check: 'second method (k-means)', ari: 0.54 }
	],
	simulated: [
		{ data: 'continuum', replicate: 1, check: '2025 forward test', ari: 0.77 },
		{ data: 'continuum', replicate: 0, check: '2025 forward test', ari: 0.56 },
		{ data: '4 classes', replicate: 0, check: '2025 forward test', ari: 0.91 }
	]
};

describe('checkRows', () => {
	it('gives every check in story order with sorted simulated values', () => {
		const rows = checkRows(replication);
		expect(rows.map((r) => r.key)).toEqual(['held-out years', '2025 forward test', 'age 6-11 refit', 'age 12-17 refit', 'second method (k-means)']);
		expect(rows[1]).toMatchObject({ real: 0.79, noGroups: [0.56, 0.77], fourGroups: [0.91] });
	});

	it('fails loudly when a check is missing', () => {
		expect(() => checkRows({ real: [], simulated: [] })).toThrow('no real result');
	});
});

describe('position', () => {
	it('places a value against a range', () => {
		expect(position(0.79, [0.56, 0.77])).toBe('above');
		expect(position(0.77, [0.85, 0.95])).toBe('below');
		expect(position(0.71, [0.66, 0.85])).toBe('within');
		expect(position(0.5411, [0.4409, 0.5408])).toBe('within');
	});
});

describe('againstRange', () => {
	it('adds decimals only when rounding would hide the difference', () => {
		expect(againstRange(0.7917, [0.5612, 0.7752])).toBe('0.79 against 0.56 to 0.78');
		expect(againstRange(0.5411, [0.4421, 0.5401])).toBe('0.541 against 0.442 to 0.540');
	});
});

describe('dominant', () => {
	it('finds the most common group and its share', () => {
		expect(dominant([2, 6, 1, 1])).toEqual({ group: 1, share: 0.6, total: 10 });
		expect(dominant([0, 0, 0, 0])).toEqual({ group: 0, share: 0, total: 0 });
	});
});

describe('sharesByGroup and shapeRanges', () => {
	it('builds one sorted series per group and the range per dataset', () => {
		const years = {
			weighted_shares: [
				{ year: 2017, n: 1, class: 'b', share: 0.4, ci_low: 0.3, ci_high: 0.5 },
				{ year: 2016, n: 1, class: 'b', share: 0.5, ci_low: 0.4, ci_high: 0.6 },
				{ year: 2016, n: 1, class: 'a', share: 0.5, ci_low: 0.4, ci_high: 0.6 }
			]
		} as unknown as Years;
		expect(sharesByGroup(years, ['a', 'b'])[1].map((p) => p.year)).toEqual([2016, 2017]);
		const shape = [
			{ data: 'real', replicate: 0, improvement: 94 },
			{ data: 'continuum', replicate: 0, improvement: 16 },
			{ data: 'continuum', replicate: 1, improvement: 22 },
			{ data: 'classes', replicate: 0, improvement: 180 }
		] as Evidence['shape'];
		expect(shapeRanges(shape)).toEqual({ real: 94, continuum: { min: 16, max: 22 }, classes: { min: 180, max: 180 } });
	});
});
