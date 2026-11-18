import { describe, expect, it } from 'vitest';
import type { Model, Profiles } from './data';
import { describeProfile, mostLikely, summarize } from './describe';
import { linear } from './scale';

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
	min_count_for_answers: 10,
	answers: ['110', null, '000'],
	count: [147, 3, 517],
	n_conditions: [2, 2, 0],
	class_probabilities: [
		[0.47, 0, 0.53, 0],
		[0.1, 0.6, 0.2, 0.1],
		[0.99, 0, 0.01, 0]
	],
	label_by_k: {},
	kmeans_label: [],
	mixture_probability: []
};

describe('describeProfile', () => {
	it('lists the conditions of a common combination, most likely group first', () => {
		const d = describeProfile(0, profiles, model);
		expect(d.sharedBy).toBe(147);
		expect(d.conditions).toEqual(['ADHD', 'Anxiety']);
		expect(d.chances.map((c) => c.name)).toEqual(['mental health and behavior', 'fewer diagnoses', 'developmental and language', 'both']);
	});

	it('never names the conditions of a rarer combination', () => {
		const d = describeProfile(1, profiles, model);
		expect(d.conditions).toBeNull();
		expect(d.sharedBy).toBeNull();
		expect(d.nConditions).toBe(2);
		const text = summarize(d);
		expect(text).toContain('rarer combination');
		for (const name of ['ADHD', 'Anxiety', 'Developmental', '3 children']) expect(text).not.toContain(name);
	});

	it('says so when a child has none of the conditions', () => {
		expect(summarize(describeProfile(2, profiles, model))).toContain('none of the 18 conditions');
	});

	it('never mentions a year or an age', () => {
		for (let i = 0; i < 3; i++) expect(summarize(describeProfile(i, profiles, model))).not.toMatch(/20\d\d|year|age \d/i);
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
