# Loom

A local-first AI coding agent. Loom indexes a Git repository, plans a change, carries it out on
a task branch, verifies it in a sandbox container, and has a second model review the result
before you decide whether to merge.

Everything runs on your own machine. Models are served by [Ollama](https://ollama.com), web
search by a local [SearXNG](https://docs.searxng.org), and code execution happens in a
throwaway container rather than on your host.

---

## How it works

You give Loom a task. It then:

1. **Indexes** the repository into Postgres — file contents, AST symbols, and vector embeddings
   for semantic search.
2. **Plans** the change and shows you the steps. Nothing happens until you approve.
3. **Executes** the plan with tools: reading and searching code, editing files, and running
   test or lint commands inside a sandbox container. Every edit is committed to a task branch.
4. **Reviews** the cumulative diff with a second model.
5. **Waits** for you to merge, or to send it back for another pass.

The run is driven by an explicit state machine, so every transition is validated and a session
can only ever be in one of a known set of states.

---

## Requirements

- **Docker** or **Podman** — for the sandbox, and for the bundled services
- **Ollama**, with at least one chat model and one embedding model pulled
- **Python 3.14+** and **Node 24+**, if you run Loom directly rather than in containers

Pull the models Loom expects by default:

```bash
ollama pull qwen2.5-coder
ollama pull nomic-embed-text
```

The embedding model must genuinely be an *embedding* model. A chat model such as `llama3` will
produce unusable vectors and semantic search will return nonsense.

---

## Quick start with Docker Compose

```bash
git clone <your-fork> loom
cd loom
docker compose up --build
```

Then open <http://localhost:5173>.

Compose brings up Postgres with pgvector, SearXNG, the backend and the frontend. It expects
Ollama at `http://ollama:11434`; change `OLLAMA_BASE_URL` in `docker-compose.yml` if yours runs
elsewhere.

Put the repository you want Loom to work on in `./data/repo`, or point `HOST_REPO_PATH` at it.

### If you use Podman

Rootless Podman keeps its socket under `XDG_RUNTIME_DIR`:

```bash
echo 'CONTAINER_SOCKET=/run/user/1000/podman/podman.sock' >> .env
docker compose up --build
```

On SELinux the socket also needs a `:Z` relabel, which cannot be expressed in a variable. Use
the override file for that:

```bash
docker compose -f docker-compose.yml -f docker-compose.podman.yml up --build
```

### If the sandbox cannot reach Docker

The backend spawns sandbox containers through the host's container socket, so the user inside
the backend container must belong to the group owning it. Find the group ID and rebuild:

```bash
stat -c '%g' /var/run/docker.sock     # e.g. 998
echo 'DOCKER_GID=998' >> .env
docker compose up --build backend
```

---

## Running directly

Useful for development, where you want reloads and a debugger.

### Services

You still need Postgres with pgvector, and SearXNG for web search:

```bash
docker compose up -d postgres searxng
```

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env        # then edit if the defaults do not suit you
uvicorn main:app --reload --port 8000
```

The schema is created by migrations at startup. There is no separate migration step.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env        # then edit if the backend is not on localhost:8000
npm run dev
```

---

## Configuration

Configuration comes from three places, in order of precedence:

1. **The environment** — a shell variable, or a value injected by Docker Compose
2. **A `.env` file** — `backend/.env` and `frontend/.env`
3. **The default** built into the code

So a value in `.env` overrides the default, and anything already in the environment overrides
the file. Docker Compose and Vite use the same ordering, so all three layers agree.

Both `.env` files are optional; a fresh clone runs without them. Copy the matching
`.env.example` to get started — it lists every variable with an explanation.

Compose reads `backend/.env` and `frontend/.env` as well, so one file serves both ways of
running. Compose then overrides the connection settings, because inside its network the
services are reachable by name rather than on `localhost`.

### Backend, `backend/.env`

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgres://loom:loom_password@localhost:5432/loom` | Postgres, with the `vector` extension available |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is listening |
| `SEARXNG_URL` | `http://localhost:8080` | Where SearXNG is listening |
| `REPO_PATH` | `/tmp/repo` | The repository, as this process sees it |
| `HOST_REPO_PATH` | *(empty)* | The same directory as the container runtime's **host** sees it. Only needed when Loom itself runs in a container. Must be absolute. |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Must be an embedding model |
| `LOOM_MAX_FILE_SIZE_BYTES` | `1048576` | Largest file indexed. `0` or negative means unlimited |
| `LOOM_MAX_PLAN_RETRIES` | `10` | Plan rejections before Loom asks how to proceed |
| `LOOM_MAX_REVIEW_RETRIES` | `10` | Review rejections before Loom asks how to proceed |
| `LOOM_MAX_TOTAL_ITERATIONS` | `20` | Hard ceiling on plan-execute-review cycles |
| `LOOM_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API |
| `AST_GREP_BIN` | `ast-grep` | Path to ast-grep, if not on `PATH` |

`HOST_REPO_PATH` deserves a note. Bind mount sources are resolved by the container runtime's
daemon, which runs on the host, so a containerised Loom asking for `/tmp/repo` would mount the
*host's* `/tmp/repo` rather than the project. This variable supplies the host's view of the
same directory. It must be absolute: a relative path is silently treated as a named volume, and
the sandbox would start with an empty tree.

### Frontend, `frontend/.env`

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Where the browser reaches the REST API |
| `VITE_WS_URL` | `ws://localhost:8000/api/v1/agent-session` | Where the browser opens the agent WebSocket |

### Compose-only variables

These configure Compose itself rather than the application. Put them in a `.env` at the
repository root, beside `docker-compose.yml`. All are optional.

| Variable | Default | Purpose |
|---|---|---|
| `HOST_REPO_PATH` | `${PWD}/data/repo` | The repository on the host |
| `DOCKER_GID` | `999` | Group that owns the container socket |
| `UID` / `GID` | `1000` | Your user, for file ownership on bind mounts |
| `CONTAINER_SOCKET` | `/var/run/docker.sock` | Path to the container runtime socket |

---

## Secrets

Secrets entered in Settings are passed to the sandbox as environment variables so the agent's
commands can use them. Their **values are never stored** — not in the database, not in browser
storage — and they are stripped from all command output before it reaches the model or the
screen. The agent is told the names only.

The consequence is that you re-enter them each browser session. That is deliberate.

---

## Tests

```bash
cd backend && pytest
cd frontend && npm test
```

The backend suite runs with no external services, skipping the parts that need them. To run
everything:

```bash
export TEST_DATABASE_URL='postgres://loom:loom_password@localhost:5432/loom'
export AST_GREP_BIN=/path/to/ast-grep     # only if not on PATH
pytest
```

`TEST_DATABASE_URL` must point at Postgres with pgvector. The suite creates and drops its own
databases, so use a development server rather than anything you care about.

Type checking:

```bash
cd backend && mypy . && basedpyright .
cd frontend && npm run check
```

---

## Project layout

```
backend/
  agents/       coder and reviewer agent definitions, and their tools
  api/v1/       HTTP routes, the WebSocket controller, and the session state machine
  database/     models, migrations, and schema setup
  services/     indexing, search, git, sandbox, chunking, and session support
  schemas/      request, response, and WebSocket message contracts
  tests/
frontend/
  src/lib/      the agent store, chat, and diff viewer
  src/routes/   the main view, settings, and task history
data/
  repo/         where Compose expects the repository being worked on
  searxng/      SearXNG configuration
```

---

## Known limitations

- **Symbol extraction covers Python, TypeScript and JavaScript.** Other languages are still
  indexed and searchable, but chunked by line window rather than on symbol boundaries.
- **One session at a time.** A second browser tab is told a session is already running.
  Browsers do not permit one tab to focus or close another, so switching is manual.
- **Sessions do not survive a backend restart.** Anything still running is marked failed at
  startup.
- **The reviewer sees a budgeted diff.** On a very large change some file diffs are summarised
  rather than included in full, and the reviewer is told which.
