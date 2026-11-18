/**
 * What the page says about one dot. This is where the publication rule lives on the
 * page side: a combination of answers shared by fewer than `min_count_for_answers`
 * children arrives without its answers, and the description then gives only how many
 * conditions it has and the group probabilities. Nothing here ever mentions a survey
 * year or an age, and the data files do not carry them.
 */
import type { Model, Profiles } from './data';
import { CONDITION_SHORT } from './groups';

export interface GroupChance {
	group: number;
	name: string;
	probability: number;
}

export interface DotDescription {
	/** How many children share these exact answers, or null for a rarer combination. */
	sharedBy: number | null;
	/** Short names of the conditions, or null when the answers are not published. */
	conditions: string[] | null;
	nConditions: number;
	/** The four groups, most likely first. */
	chances: GroupChance[];
}

export function describeProfile(index: number, profiles: Profiles, model: Model): DotDescription {
	const answers = profiles.answers[index];
	const chances = profiles.class_probabilities[index]
		.map((probability, group) => ({ group, name: model.classes[group].short_name, probability }))
		.sort((a, b) => b.probability - a.probability);
	if (answers === null) {
		return { sharedBy: null, conditions: null, nConditions: profiles.n_conditions[index], chances };
	}
	const conditions = model.conditions.filter((_, j) => answers[j] === '1').map((c) => CONDITION_SHORT[c.flag] ?? c.label);
	return { sharedBy: profiles.count[index], conditions, nConditions: conditions.length, chances };
}

/** The group a profile is drawn in, and whether the model is confident about it. */
export function mostLikely(index: number, profiles: Profiles): { group: number; probability: number } {
	const row = profiles.class_probabilities[index];
	let group = 0;
	for (let g = 1; g < row.length; g++) if (row[g] > row[group]) group = g;
	return { group, probability: row[group] };
}

/** One sentence for screen readers and the tooltip's first line. */
export function summarize(d: DotDescription): string {
	const top = d.chances[0];
	const where = `most likely in the ${top.name} group (${Math.round(100 * top.probability)}%)`;
	if (d.conditions === null) {
		return `A rarer combination of answers with ${d.nConditions} of the 18 conditions, ${where}.`;
	}
	const what = d.conditions.length === 0 ? 'none of the 18 conditions' : d.conditions.join(', ');
	return `${d.sharedBy} children share these answers: ${what}; ${where}.`;
}
