/**
 * The shapes of the JSON files in static/data, written by scripts/build_site_data.py,
 * and one loader that respects the GitHub Pages subpath.
 */
import { asset } from '$app/paths';

/** One entry per child, as columns, in a shuffled order. */
export interface Children {
	n: number;
	/** Position on the developmental-language axis, drawn under the model's bell curve. */
	x: number[];
	/** Position on the mental-health axis, drawn under the model's bell curve. */
	y: number[];
	sharp_x: number[];
	sharp_y: number[];
	/** Row in Profiles. */
	profile: number[];
}

/** One entry per distinct set of 18 answers, most common first. */
export interface Profiles {
	/** "0"/"1" per condition, in model order. */
	answers: string[];
	count: number[];
	n_conditions: number[];
	/** Probability of each of the four groups, in site order. */
	class_probabilities: number[][];
	label_by_k: Record<string, number[]>;
	kmeans_label: number[];
	mixture_probability: number[];
}

export interface Condition {
	flag: string;
	label: string;
	prevalence: number;
}

export interface Group {
	name: string;
	short_name: string;
	share_of_children: number;
	weighted_share: number;
	rates: number[];
	center: [number, number];
}

export interface Model {
	conditions: Condition[];
	axes: { x: { name: string; markers: string[] }; y: { name: string; markers: string[] } };
	classes: Group[];
	by_k: { k: number; color_slot: number[]; shares: number[]; centers: [number, number][] }[];
}

export type Numbers = Record<string, { value: number; source: string }>;

/** The files make site-data writes; SvelteKit checks each against static/data at compile time. */
export type DataFile =
	| 'children.json'
	| 'profiles.json'
	| 'model.json'
	| 'clouds.json'
	| 'evidence.json'
	| 'outcomes.json'
	| 'years.json'
	| 'context.json'
	| 'numbers.json';

/** Fetch one of the site's data files. */
export async function loadData<T>(name: DataFile, fetcher: typeof fetch = fetch): Promise<T> {
	const response = await fetcher(asset(`data/${name}`));
	if (!response.ok) throw new Error(`Could not load ${name}: the server answered ${response.status}.`);
	return (await response.json()) as T;
}
