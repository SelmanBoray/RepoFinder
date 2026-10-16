/** Geometry helpers. */
export namespace Geometry {
  export const ORIGIN = 0;

  export function distance(a: number, b: number): number {
    return Math.abs(a - b);
  }

  export namespace Shapes {
    export class Circle {
      area(): number {
        return 0;
      }
    }

    export interface Sized {
      size: number;
    }
  }
}

namespace App.Config {
  export function load(): void {}
}

module Legacy {
  function helper() {}
}

declare module "external-lib" {
  export function patched(): void;
}

declare global {
  interface Window {
    appVersion: string;
  }
}
