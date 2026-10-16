import { Injectable, Input } from "./decorators";

/** A service with every kind of member. */
@Injectable()
export class UserService extends BaseService implements Disposable {
  static instances = 0;
  private cache = new Map<string, User>();

  constructor(private readonly http: Http) {
    super();
  }

  /** Current user name. */
  get name(): string {
    return this.http.user;
  }

  set name(value: string) {
    this.http.user = value;
  }

  /** Load a user. */
  @Input()
  @Cached({ ttl: 5 })
  async load(id: string): Promise<User> {
    const parse = (raw: string) => JSON.parse(raw);
    return parse(await this.http.get(id));
  }

  static create(): UserService {
    return new UserService(new Http());
  }

  /** Arrow property keeps `this`. */
  onClick = (event: MouseEvent): void => {
    this.load(String(event.button));
  };

  #secret(): number {
    return 42;
  }

  ["literal-name"](): void {}

  [Symbol.iterator](): Iterator<User> {
    return [].values();
  }
}

export abstract class Shape {
  abstract area(): number;
  describe(): string {
    return `area ${this.area()}`;
  }
}

export const Point = class {
  norm(): number {
    return 0;
  }
};
