import defaultThing from "./a";
import { alpha, beta as b } from "../shared/b.js";
import * as ns from "some-pkg";
import type { TypeOnly } from "./types";
import Def, { type Named, other } from "./lib";
import "./polyfill";
import legacy = require("./legacy");
export { reexported } from "./y";
export * from "./z";
export * as zz from "./zz.mjs";
export { default as Widget, helper } from "./widgets";
const cjs = require("./cjs-module");
const { pick, omit: drop } = require("lodash");

export async function lazy(): Promise<void> {
  const mod = await import("./lazy-mod.js");
  const { feature } = await import("./features");
  const dynamic = await import(`./locales/${lang}.js`);
  void import("./side-effect");
}
