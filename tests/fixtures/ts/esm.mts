import { readFile } from "node:fs/promises";
import { helper } from "./helper.mjs";

/** Read a config file. */
export async function readConfig(path: string): Promise<string> {
  return helper(await readFile(path, "utf8"));
}

export class Loader {
  load = async (path: string) => readConfig(path);
}
