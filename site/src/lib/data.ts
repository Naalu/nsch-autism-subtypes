/**
 * The shapes of the JSON files in static/data, written by scripts/build_site_data.py,
 * and one loader that respects the GitHub Pages subpath.
 */
import { base } from '$app/paths';

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
	min_count_for_answers: number;
	/** "0"/"1" per condition in model order; null when fewer than min_count_for_answers children share it. */
	answers: (string | null)[];
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

/** Fetch one of the site's data files. */
export async function loadData<T>(name: string, fetcher: typeof fetch = fetch): Promise<T> {
	const response = await fetcher(`${base}/data/${name}`);
	if (!response.ok) throw new Error(`Could not load ${name}: the server answered ${response.status}.`);
	return (await response.json()) as T;
}
