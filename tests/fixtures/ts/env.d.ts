declare function setup(options?: object): void;
declare function setup(name: string, options?: object): void;

declare const VERSION: string;

declare namespace Env {
  function get(key: string): string | undefined;
}
