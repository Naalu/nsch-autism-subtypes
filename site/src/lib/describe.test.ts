import { describe, expect, it } from 'vitest';
import type { Model, Profiles } from './data.ts';
import { abbreviate, describeProfile, diagnosesText, mostLikely, sharedText, summarize } from './describe.ts';
import { inSentence } from './groups.ts';
import { linear } from './scale.ts';

const model = {
	conditions: [
		{ flag: 'adhd', label: 'ADD/ADHD', prevalence: 0.5 },
		{ flag: 'anxiety', label: 'Anxiety problems', prevalence: 0.5 },
		{ flag: 'down_syndrome', label: 'Down syndrome', prevalence: 0.01 }
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
	it('lists the diagnoses as they read in a sentence, and the groups best fit first', () => {
		const d = describeProfile(1, profiles, model);
		expect(d.diagnoses).toEqual(['ADHD', 'anxiety', 'Down syndrome']);
		expect(describeProfile(0, profiles, model).chances.map((c) => c.name)).toEqual([
			'mental health and behavior',
			'fewer diagnoses',
			'developmental and language',
			'both'
		]);
	});

	it('ties the group to the answers, never to the child, and never mentions a year or an age', () => {
		for (let i = 0; i < 3; i++) {
			const text = summarize(describeProfile(i, profiles, model), 8891);
			expect(text).toContain("best fit for these answers");
			expect(text).not.toMatch(/20\d\d|year|age \d|this child is/i);
		}
	});
});

describe('diagnosesText', () => {
	it('shortens long lists and can show them in full', () => {
		const five = ['ADHD', 'anxiety', 'depression', 'behavior problems', 'asthma'];
		expect(diagnosesText(five)).toBe('5 diagnoses: ADHD, anxiety, depression, and 2 more');
		expect(diagnosesText(five, true)).toBe('5 diagnoses: ADHD, anxiety, depression, behavior problems, asthma');
		expect(diagnosesText(['asthma'])).toBe('1 diagnosis: asthma');
		expect(diagnosesText([])).toBe('None of the 18 diagnoses');
	});
});

describe('sharedText', () => {
	it('counts within the children studied, not the whole survey', () => {
		expect(sharedText(1, 8891)).toBe('1 of the 8,891 children has exactly these answers.');
		expect(sharedText(147, 8891)).toBe('147 of the 8,891 children have exactly these answers.');
	});
});

describe('helpers', () => {
	it('abbreviates, finds the best fit, lowercases names and maps scales', () => {
		expect(abbreviate([1, 2, 3, 4])).toEqual({ first: [1, 2, 3], more: 1 });
		expect(mostLikely(0, profiles)).toEqual({ group: 2, probability: 0.53 });
		expect(['ADHD', 'Down syndrome', 'Tourette syndrome', 'Speech or language disorder'].map(inSentence)).toEqual([
			'ADHD',
			'Down syndrome',
			'Tourette syndrome',
			'speech or language disorder'
		]);
		const s = linear([-4, 4], [0, 800]);
		expect([s(0), s.invert(200)]).toEqual([400, -2]);
	});
});
