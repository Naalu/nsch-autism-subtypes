import { describe, expect, it } from 'vitest';
import type { Model, Profiles } from './data.ts';
import { abbreviate, describeProfile, mostLikely, sharedText, summarize } from './describe.ts';
import { linear } from './scale.ts';

const model = {
	conditions: [
		{ flag: 'adhd', label: 'ADD/ADHD', prevalence: 0.5 },
		{ flag: 'anxiety', label: 'Anxiety problems', prevalence: 0.5 },
		{ flag: 'dev_delay', label: 'Developmental delay', prevalence: 0.5 }
	],
	classes: ['fewer diagnoses', 'developmental and language', 'mental health and behavior', 'both'].map((short_name) => ({
		name: short_name,
		short_name,
		share_of_children: 0.25,
		weighted_share: 0.25,
		rates: [],
		center: [0, 0]
	}))
} as unknown as Model;

const profiles: Profiles = {
	answers: ['110', '111', '000'],
	count: [147, 1, 517],
	n_conditions: [2, 3, 0],
	class_probabilities: [
		[0.47, 0, 0.53, 0],
		[0.1, 0.2, 0.1, 0.6],
		[0.99, 0, 0.01, 0]
	],
	label_by_k: {},
	kmeans_label: [],
	mixture_probability: []
};

describe('describeProfile', () => {
	it('lists the conditions in survey order and the groups most likely first', () => {
		const d = describeProfile(0, profiles, model);
		expect(d.sharedBy).toBe(147);
		expect(d.conditions).toEqual(['ADHD', 'Anxiety']);
		expect(d.chances.map((c) => c.name)).toEqual(['mental health and behavior', 'fewer diagnoses', 'developmental and language', 'both']);
	});

	it('says so when a child has none of the conditions', () => {
		expect(summarize(describeProfile(2, profiles, model))).toContain('none of the 18 conditions');
	});

	it('never mentions a year or an age', () => {
		for (let i = 0; i < 3; i++) expect(summarize(describeProfile(i, profiles, model))).not.toMatch(/20\d\d|year|age \d/i);
	});
});

describe('abbreviate', () => {
	it('keeps the first few and counts the rest', () => {
		expect(abbreviate(['a', 'b', 'c', 'd', 'e'])).toEqual({ first: ['a', 'b', 'c'], more: 2 });
		expect(abbreviate(['a', 'b'])).toEqual({ first: ['a', 'b'], more: 0 });
	});
});

describe('sharedText', () => {
	it('words a set of answers held by one child differently', () => {
		expect(sharedText(1)).toContain('No other child');
		expect(sharedText(1234)).toBe('1,234 children share exactly these answers.');
	});
});

describe('mostLikely', () => {
	it('returns the largest probability', () => {
		expect(mostLikely(0, profiles)).toEqual({ group: 2, probability: 0.53 });
	});
});

describe('linear', () => {
	it('maps and inverts', () => {
		const s = linear([-4, 4], [0, 800]);
		expect(s(0)).toBe(400);
		expect(s.invert(200)).toBe(-2);
	});
});
