/** A linear map from one interval to another, and its inverse. */
export function linear(domain: [number, number], range: [number, number]) {
	const [d0, d1] = domain;
	const [r0, r1] = range;
	const k = (r1 - r0) / (d1 - d0);
	const map = (v: number) => r0 + (v - d0) * k;
	map.invert = (r: number) => d0 + (r - r0) / k;
	return map;
}
