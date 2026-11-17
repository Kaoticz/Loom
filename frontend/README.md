# Loom Frontend

Loom is a local agentic coder assistant. This is the frontend component, built with SvelteKit, TypeScript and TailwindCSS.

## Setup

### Prerequisites

- Node.js 25.x or later
- npm

### Installation

1. Install the dependencies:

```bash
npm install
```

2. Configure the environment variables:

```bash
cp env.example .env
```

Edit `.env` to point `VITE_API_BASE_URL` and `VITE_WS_URL` at the backend. Only variables prefixed with `VITE_` are exposed to client code, and anything already set in the shell, or injected by Docker Compose, takes precedence over the file.

### Backend Dependency

The frontend expects the Loom backend to be running, as the agent chat is driven entirely by its HTTP API and WebSocket endpoint. See `backend/README.md` for instructions on starting it.

## Maintenance

### Static Analysis

Run `svelte-check` to check for type errors:

```bash
npm run check
```

Run Prettier and ESLint to check for formatting and lint issues:

```bash
npm run lint
```

### Tests

Run the test suite with Vitest:

```bash
npm run test
```

A handful of scrolling behaviours can only be checked against a real browser, because jsdom performs no layout. Those tests skip unless `LOOM_BROWSER_BIN` is set in `.env`, pointing at a Chromium-family executable.

### Starting the Dev Server

Run the SvelteKit application with Vite:

```bash
npm run dev
```

## Dependencies

- **SvelteKit**: Web framework.
- **Svelte 5**: UI framework.
- **TailwindCSS**: Utility-first styling.
- **TypeScript**: Static typing.
- **Vite**: Dev server and bundler.
- **Zod**: Runtime validation of data crossing trust boundaries.
- **CodeMirror**: Code editor, used by the diff viewer.
- **marked**: Markdown rendering.
- **DOMPurify**: Sanitization of rendered Markdown.
- **Vitest**: Test runner.
