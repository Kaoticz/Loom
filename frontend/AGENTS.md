# AGENTS.md

This file describes the project's stack, conventions, and code standards for AI agents working in this codebase.

---

## Stack

- **Framework:** SvelteKit
- **Language:** TypeScript
- **Styling:** TailwindCSS
- **WebSockets:** Native browser WebSocket API via Svelte stores

---

## Prerequisites

- Node.js **25.x** or later is required. Do not assume an older version.
- Package manager: **npm**.

---

## Programming Style

The codebase follows a **functional, reactive** style idiomatic to Svelte. When writing code:

- Use Svelte stores to model shared, reactive state — particularly for WebSocket-backed data streams.
- Use pure functions for logic that transforms data without side effects.
- Co-locate component logic, markup, and styles within `.svelte` files unless the logic is reusable, in which case extract it to a `.ts` module.
- Do not default to class-based patterns; reserve them for cases where encapsulation of stateful behaviour genuinely benefits from it.

---

## Code Standards

These rules are **mandatory**. Apply them to every file you create or modify.

### Type Annotations

Always annotate types wherever possible — function signatures, return types, and local variables. Enable `strict` mode in `tsconfig.json` and treat type errors as build failures.

```typescript
// Good
function getUser(userId: number): User {
  const result: User | undefined = repository.find(userId);
  ...
}

// Bad
function getUser(userId) {
  const result = repository.find(userId);
  ...
}
```

### JSDoc Comments

Every exported function, store, and class must have a JSDoc comment. Include:

- A short summary on the first line.
- `@param` for each parameter.
- `@returns` describing the return value (if any).
- `@throws` for each exception the function may raise (if any).

```typescript
/**
 * Create a new WebSocket connection for the given session.
 *
 * @param sessionId - The ID of the agent session to connect to.
 * @returns A readable store that emits incoming messages.
 * @throws {ConnectionError} If the WebSocket handshake fails.
 */
export function createSessionSocket(sessionId: string): Readable<AgentMessage> {
  ...
}
```

### Quotes

Prefer **single quotes** (`'`) for strings over double quotes (`"`), except in template literals and JSX/HTML attribute values where double quotes are standard.

```typescript
// Good
const name: string = 'Alice';

// Bad
const name: string = 'Alice';
```

### API Contracts — Request & Response Types

Every API call must be backed by dedicated TypeScript types for the request payload and the response shape. Do not use raw `any`, untyped `fetch` responses, or generic `object` types as API contracts.

```typescript
interface CreateRunRequest {
  prompt: string;
  modelId: string;
}

interface CreateRunResponse {
  id: string;
  status: RunStatus;
  createdAt: string;
}

async function createRun(body: CreateRunRequest): Promise<CreateRunResponse> {
  ...
}
```

### Zod for Runtime Validation

Use **Zod** to validate all data that crosses a trust boundary — API responses, WebSocket messages, and environment variables. Never assume external data conforms to a TypeScript type without runtime validation.

```typescript
// Good
const AgentMessageSchema = z.object({
    type: z.enum(['token', 'error', 'done']),
    content: z.string()
});

const message: AgentMessage = AgentMessageSchema.parse(raw);

// Bad
const message = raw as AgentMessage;
```

### Enums for Literal Types

When a variable or type field can take **3 or more** named literal values, define a string union type or a `const` object rather than scattering inline string literals.

```typescript
// Good
const RunStatus = {
    pending: 'pending',
    running: 'running',
    completed: 'completed',
    failed: 'failed'
} as const;

type RunStatus = (typeof RunStatus)[keyof typeof RunStatus];

// Bad
type Run = {
    status: 'pending' | 'running' | 'completed' | 'failed';
};
```

### Guard Clauses

Prefer early returns to reduce nesting and clarify intent.

```typescript
// Good
function processMessage(message: AgentMessage): void {
    if (message.type === 'error') {
        handleError(message.content);
        return;
    }
    if (!message.content) {
        return;
    }
    appendToken(message.content);
}

// Bad
function processMessage(message: AgentMessage): void {
    if (message.type !== 'error') {
        if (message.content) {
            appendToken(message.content);
        }
    } else {
        handleError(message.content);
    }
}
```

### No Side Effects

Functions should return values rather than mutating external state silently. Svelte store updates are the sanctioned mechanism for reactive state changes — keep them explicit and localised.

```typescript
// Good
function withStatus(run: Run, status: RunStatus): Run {
    return { ...run, status };
}

// Bad
function setStatus(run: Run, status: RunStatus): void {
    run.status = status;
}
```

### WebSocket Stores

All WebSocket connections must be encapsulated in a Svelte store factory. Components must never instantiate `WebSocket` directly. The store is responsible for connection lifecycle, reconnection logic, and message parsing.

```typescript
// Good — in src/lib/stores/agentSocket.ts
export function createAgentSocket(sessionId: string): Readable<AgentMessage | null> {
  return readable<AgentMessage | null>(null, (set) => {
    const ws = new WebSocket(`/ws/sessions/${sessionId}`);

    ws.onmessage = (event: MessageEvent) => {
      const message = AgentMessageSchema.parse(JSON.parse(event.data));
      set(message);
    };

    return () => ws.close();
  });
}

// Bad — inside a .svelte component
const ws = new WebSocket(`/ws/sessions/${sessionId}`);
ws.onmessage = (e) => { ... };
```

### Component Responsibilities

Each Svelte component must have a single, clearly defined responsibility. If a component file grows beyond ~200 lines, extract logic into a `.ts` module or split into sub-components.

- Keep data-fetching and store subscriptions in page-level components or `+page.svelte` files.
- Keep presentational components free of direct API calls or store mutations.
- Pass data down via props; communicate upward via dispatched events or callback props.

### Post-Task

Once you're done writing your changes, execute the following commands:

```bash
npm run check
```

```bash
npm run lint
```

Fix any error or warning returned by the linters, even if they are not related to the task at hand.

---

## Common Pitfalls to Avoid

- Do not use `any` as an escape hatch — use `unknown` and narrow the type explicitly.
- Do not instantiate `WebSocket` directly inside components.
- Do not share API response types across call sites if their shapes diverge.
- Do not mutate store values directly — always use the store's `set` or `update` methods.
- Do not leave Zod validation absent at trust boundaries, even when the data source seems reliable.

---

## General Agent Guidance

- Do not introduce new dependencies without being asked.
- Do not modify the SvelteKit routing structure without being asked.
- When adding a new API call, always create accompanying request and response types.
- When in doubt about naming or structure, follow the patterns already established in the codebase.
