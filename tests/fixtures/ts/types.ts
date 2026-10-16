/** Identifier of a user. */
export type UserId = string;

export type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

/**
 * A user record.
 */
export interface User extends Entity<UserId> {
  id: UserId;
  name: string;
  greet(other: User): string;
}

interface Internal {}

/** Log levels. */
export enum Level {
  Debug,
  Info = 2,
}

export const enum Flag {
  On = 1,
}

declare function legacyGlobal(x: number): void;

export declare class Remote {
  connect(url: string): Promise<void>;
  connect(url: URL): Promise<void>;
  close(): void;
}
