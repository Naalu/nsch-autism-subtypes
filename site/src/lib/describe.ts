/**
 * What the page says about one dot: the diagnoses in its set of answers, how many of the
 * children share that set, and the four-group model's best fit for it. The wording keeps
 * the fit attached to the answers, not to the child. Nothing here mentions a survey year
 * or an age, and the data files do not carry them.
 */
import type { Model, Profiles } from './data.ts';
import { CONDITION_SHORT, inSentence } from './groups.ts';

export interface GroupChance {
	group: number;
	name: string;
	probability: number;
}

export interface DotDescription {
	/** How many children share exactly these answers. */
	sharedBy: number;
	/** Short names of the diagnoses as they read in a sentence, in survey order. */
	diagnoses: string[];
	/** The four groups, best fit first. */
	chances: GroupChance[];
}

export function describeProfile(index: number, profiles: Profiles, model: Model): DotDescription {
	const answers = profiles.answers[index];
	const diagnoses = model.conditions.filter((_, j) => answers[j] === '1').map((c) => inSentence(CONDITION_SHORT[c.flag] ?? c.label));
	const chances = profiles.class_probabilities[index]
		.map((probability, group) => ({ group, name: model.classes[group].short_name, probability }))
		.sort((a, b) => b.probability - a.probability);
	return { sharedBy: profiles.count[index], diagnoses, chances };
}

/** The first `shown` items and how many more there are. */
export function abbreviate<T>(items: T[], shown = 3): { first: T[]; more: number } {
	return { first: items.slice(0, shown), more: Math.max(0, items.length - shown) };
}

/** "9 diagnoses: ADHD, anxiety, depression, and 6 more", or the full list. */
export function diagnosesText(diagnoses: string[], full = false): string {
	if (diagnoses.length === 0) return 'None of the 18 diagnoses';
	const head = `${diagnoses.length} ${diagnoses.length === 1 ? 'diagnosis' : 'diagnoses'}`;
	const { first, more } = abbreviate(diagnoses);
	return full || more === 0 ? `${head}: ${diagnoses.join(', ')}` : `${head}: ${first.join(', ')}, and ${more} more`;
}

/** How many of the children share a set of answers, in words. */
export function sharedText(sharedBy: number, total: number): string {
	const of = `of the ${total.toLocaleString('en-US')} children`;
	return sharedBy === 1 ? `1 ${of} has exactly these answers.` : `${sharedBy.toLocaleString('en-US')} ${of} have exactly these answers.`;
}

/** The group a profile is drawn in, and the model's probability for it. */
export function mostLikely(index: number, profiles: Profiles): { group: number; probability: number } {
	const row = profiles.class_probabilities[index];
	let group = 0;
	for (let g = 1; g < row.length; g++) if (row[g] > row[group]) group = g;
	return { group, probability: row[group] };
}

/** One sentence for screen readers. */
export function summarize(d: DotDescription, total: number): string {
	const top = d.chances[0];
	return `${diagnosesText(d.diagnoses, true)}. ${sharedText(d.sharedBy, total)} The four-group model's best fit for these answers is ${top.name} (${Math.round(100 * top.probability)}%).`;
}
