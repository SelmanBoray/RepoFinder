/**
 * Utilities for the demo module.
 */

/**
 * Add two numbers.
 *
 * The second paragraph is not part of the summary.
 * @param a first
 */
export function add(a: number, b: number): number {
  return a + b;
}

/** Fetch a user by id. */
export async function fetchUser(
  id: string, // the user id
  opts?: { cache: boolean },
): Promise<User | undefined> {
  function normalize(raw: string): string {
    const trim = (s: string) => s.trim();
    return trim(raw);
  }
  return lookup(normalize(id), opts);
}

export function* counter(): Generator<number> {
  yield 1;
}

/** Doubles a value. */
export const double = (x: number): number => x * 2;

/** Legacy wrapper. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const legacy = function legacyImpl(value: any) {
  return value;
};

/* A plain block comment is not JSDoc. */
let handler = async ({ id }: { id: string }) => {
  await fetchUser(id);
};

const LIMIT = 10;

export function parse(input: string): number;
export function parse(input: number): number;
export function parse(input: string | number): number {
  return Number(input);
}

if (process.env.DEBUG) {
  function debugLog(message: string): void {
    console.log(message);
  }
}

/** Too far away to document anything. */


function detached() {}

var first = () => 1,
  second = function () {
    return 2;
  };
