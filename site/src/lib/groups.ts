/**
 * Colors and labels. The four group colors mean "this group" everywhere on the site and
 * nothing else. They were checked with the dataviz palette validator against the page
 * background: every pair separates under protan, deutan and tritan simulation (worst
 * ΔE 8.2) and for normal vision (worst 15.8). The gray is deliberately a neutral, for the
 * group with fewer diagnoses, and is below 3:1 contrast, so groups are always labeled in
 * text as well.
 */
export const GROUP_COLORS = ['#969ba8', '#0b78a8', '#6b46c1', '#bf5a2a'] as const;

/** Ink for "has this condition" when the map is colored by one condition. */
export const INK = '#1c2230';
/** Faint dot: "does not have this condition", or a combination whose answers are not shown. */
export const FAINT = '#c3c9d2';

/** A child is drawn faded when no group has at least this probability. */
export const CONFIDENT = 0.8;

/** Short names for the 18 conditions, for compact labels. Keys are the survey flags. */
export const CONDITION_SHORT: Record<string, string> = {
	adhd: 'ADHD',
	anxiety: 'Anxiety',
	depression: 'Depression',
	behavior: 'Behavior problems',
	dev_delay: 'Developmental delay',
	intellectual: 'Intellectual disability',
	speech: 'Speech or language disorder',
	learning: 'Learning disability',
	tourette: 'Tourette syndrome',
	asthma: 'Asthma',
	epilepsy: 'Epilepsy',
	headache: 'Headaches',
	heart: 'Heart condition',
	cerebral_palsy: 'Cerebral palsy',
	blood: 'Blood disorder',
	down_syndrome: 'Down syndrome',
	deafness: 'Hearing problems',
	blindness: 'Vision problems'
};

/** The three families the conditions are shown in, in survey order. */
export const CONDITION_FAMILIES = [
	{ name: 'Mental health and behavior', flags: ['adhd', 'anxiety', 'depression', 'behavior'] },
	{ name: 'Developmental and language', flags: ['dev_delay', 'intellectual', 'speech', 'learning'] },
	{
		name: 'Other conditions',
		flags: ['tourette', 'asthma', 'epilepsy', 'headache', 'heart', 'cerebral_palsy', 'blood', 'down_syndrome', 'deafness', 'blindness']
	}
] as const;

/** A condition's short name as it reads inside a sentence: lowercase unless it starts with a name or acronym. */
export function inSentence(name: string): string {
	return /^(ADHD|Down|Tourette)/.test(name) ? name : name.charAt(0).toLowerCase() + name.slice(1);
}

export function percent(p: number, digits = 0): string {
	return `${(100 * p).toFixed(digits)}%`;
}
