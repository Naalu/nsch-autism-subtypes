/**
 * What the page says about one dot: the conditions in its set of answers, how many
 * children share that set, and the group probabilities. Nothing here mentions a survey
 * year or an age, and the data files do not carry them.
 */
import type { Model, Profiles } from './data.ts';
import { CONDITION_SHORT } from './groups.ts';

export interface GroupChance {
	group: number;
	name: string;
	probability: number;
}

export interface DotDescription {
	/** How many children share exactly these answers. */
	sharedBy: number;
	/** Short names of the conditions, in survey order. */
	conditions: string[];
	/** The four groups, most likely first. */
	chances: GroupChance[];
}

export function describeProfile(index: number, profiles: Profiles, model: Model): DotDescription {
	const answers = profiles.answers[index];
	const conditions = model.conditions.filter((_, j) => answers[j] === '1').map((c) => CONDITION_SHORT[c.flag] ?? c.label);
	const chances = profiles.class_probabilities[index]
		.map((probability, group) => ({ group, name: model.classes[group].short_name, probability }))
		.sort((a, b) => b.probability - a.probability);
	return { sharedBy: profiles.count[index], conditions, chances };
}

/** The first `shown` conditions and how many more there are. */
export function abbreviate(conditions: string[], shown = 3): { first: string[]; more: number } {
	return { first: conditions.slice(0, shown), more: Math.max(0, conditions.length - shown) };
}

/** Who shares a set of answers, in words. */
export function sharedText(sharedBy: number): string {
	return sharedBy === 1 ? 'No other child in the survey has exactly these answers.' : `${sharedBy.toLocaleString('en-US')} children share exactly these answers.`;
}

/** The group a profile is drawn in, and whether the model is confident about it. */
export function mostLikely(index: number, profiles: Profiles): { group: number; probability: number } {
	const row = profiles.class_probabilities[index];
	let group = 0;
	for (let g = 1; g < row.length; g++) if (row[g] > row[group]) group = g;
	return { group, probability: row[group] };
}

/** One sentence for screen readers. */
export function summarize(d: DotDescription): string {
	const top = d.chances[0];
	const what = d.conditions.length === 0 ? 'none of the 18 conditions' : `${d.conditions.length} conditions: ${d.conditions.join(', ')}`;
	return `${what}. ${sharedText(d.sharedBy)} Most likely in the ${top.name} group (${Math.round(100 * top.probability)}%).`;
}
