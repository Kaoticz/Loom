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

1. **Indexes** the repository into Postgres, file contents, AST symbols, and vector embeddings
   for semantic search.
2. **Classifies** the task. If it only calls for an answer, explaining code, assessing a design,
   Loom answers it and stops, with no plan, branch, or review to sit through.
3. **Plans** the change and shows you the steps. Nothing happens until you approve.
4. **Executes** the plan with tools: reading and searching code, editing files, and running
   test or lint commands inside a sandbox container. Every edit is committed to a task branch.
5. **Reviews** the cumulative diff with a second model.
6. **Waits** for you to merge, or to send it back for another pass.

---

## Requirements

- **Docker** or **Podman**, for the sandbox and for the bundled services
- **Ollama**, with at least one chat model and one embedding model pulled
- **Python 3.14+** and **Node 25+**, if you run Loom directly rather than in containers

Pull the models Loom expects by default:

```bash
ollama pull qwen3.5:9b  # Any LLM that supports tool calls will do
ollama pull snowflake-arctic-embed2  # The default embedding model
```

If you would rather embed with a different model, such as `nomic-embed-text`, set
`EMBEDDING_MODEL` to match whatever you pulled.

---

## Quick start with Docker Compose

```bash
git clone https://github.com/Kaoticz/Loom
cd Loom
docker compose up --build
```

Then open <http://localhost:5173>.

Compose brings up Postgres with pgvector, Ollama, SearXNG, the backend and the frontend. The
bundled Ollama shares the host's network, and the backend is pointed at
`http://host.docker.internal:11434`. If yours listens elsewhere, change `OLLAMA_BASE_URL`
in `docker-compose.yml`.

The bundled Ollama keeps its pulled models in `./data/ollama` and reserves every NVIDIA GPU it
can see, which asks for the NVIDIA container toolkit. On a machine without one, remove the
`deploy` block from the `ollama` service in `docker-compose.yml`.

Put the repository you want Loom to work on in `./data/repo` (a symlink also works), then pick it
with the directory picker in the interface. Anything beneath a directory mounted into the Loom
container will do.

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

`dev.sh` does the setup for you. It creates the virtualenv, installs everything including the
test and type-checking tools, and defines a `migrate` helper around Aerich:

```bash
cd backend
source dev.sh
fastapi dev
```

Driven by hand instead:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

cp env.example .env         # then edit if the defaults do not suit you
uvicorn main:app --reload --port 8000
```

The schema is created by Aerich migrations, applied automatically at startup. There is no
separate migration step. After the models change, `migrate add <name>` writes the migration and
`migrate apply` takes it to the database.

### Frontend

```bash
cd frontend
npm install
cp env.example .env         # then edit if the backend is not on localhost:8000
npm run dev
```

---

## Configuration

Configuration comes from three places, in order of precedence:

1. **The environment**, a shell variable, or a value injected by Docker Compose
2. **A `.env` file**, `backend/.env` and `frontend/.env`
3. **The default** built into the code

So a value in `.env` overrides the default, and anything already in the environment overrides
the file. Docker Compose and Vite use the same ordering, so all three layers agree.

Both `.env` files are optional. A fresh clone runs without them. Copy the matching `env.example`
to get started, it lists every variable with an explanation.

Compose reads `backend/.env` and `frontend/.env` as well, so one file serves both ways of
running. Compose then overrides the connection settings, because inside its network the
services are reachable by name rather than on `localhost`.

### Backend, `backend/.env`

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgres://loom:loom_password@localhost:5432/loom_db` | Postgres, with the `vector` extension available |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is listening |
| `SEARXNG_URL` | `http://localhost:8080` | Where SearXNG is listening |
| `REPO_PATH` | `/tmp/repo` | The repository, as this process sees it |
| `EMBEDDING_MODEL` | `snowflake-arctic-embed2` | Must be an embedding model |
| `LOOM_MAX_FILE_SIZE_BYTES` | `1048576` | Largest file indexed. `0` or negative means unlimited |
| `LOOM_MAX_PLAN_RETRIES` | `10` | Plan rejections before Loom asks how to proceed |
| `LOOM_MAX_REVIEW_RETRIES` | `10` | Review rejections before Loom asks how to proceed |
| `LOOM_MAX_TOTAL_ITERATIONS` | `20` | Hard ceiling on plan-execute-review cycles |
| `LOOM_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API |
| `AST_GREP_BIN` | `ast-grep` | Path to ast-grep, if not on `PATH` |

Bind mount sources are resolved by the container runtime's daemon, which runs on the host, so a
containerised Loom asking for `/tmp/repo` would mount the *host's* `/tmp/repo` rather than the
project. Loom discovers the host's view of the repository by inspecting its own container's mounts,
so nothing needs configuring. If it cannot, because the repository is not under anything mounted
in from the host, the sandbox is reported as unavailable and the session continues without being
able to verify its work.

### Frontend, `frontend/.env`

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Where the browser reaches the REST API |
| `VITE_WS_URL` | `ws://localhost:8000/api/v1/agent-session` | Where the browser opens the agent WebSocket |

### Compose-only variables

These configure Compose itself rather than the application. Put them in a `.env` at the
repository root, beside `docker-compose.yml`. The `env.example` there is a template for it,
and also carries the test suite's `TEST_DATABASE_URL`. All are optional.

| Variable | Default | Purpose |
|---|---|---|
| `DOCKER_GID` | `999` | Group that owns the container socket |
| `UID` / `GID` | `1000` | Your user, for file ownership on bind mounts |
| `CONTAINER_SOCKET` | `/var/run/docker.sock` | Path to the container runtime socket |

---

## Secrets

Secrets entered in Settings are passed to the sandbox as environment variables so the agent's
commands can use them. Their **values are never stored** and they are stripped from all command
output before it reaches the model or the screen. The agent is told the names only.

This means you have re-enter them each browser session. This is deliberate.

---

## Tests

```bash
cd backend && pytest
cd frontend && npm test
```

The backend suite runs with no external services, skipping the parts that need them. It requires
its dev extras installed first (`pip install -e '.[dev]'`, or `source dev.sh`). To run
everything:

```bash
export TEST_DATABASE_URL='postgres://loom:loom_password@localhost:5432/loom_db'
export AST_GREP_BIN=/path/to/ast-grep     # only if not on PATH
pytest
```

`TEST_DATABASE_URL` must point at Postgres with pgvector. The suite creates and drops its own
databases, so use a development server rather than anything you care about.

The driver, `playwright-core`, is an optional dependency that downloads no browsers
of its own. Each of these variables is explained in an `env.example` file: `TEST_DATABASE_URL`
in the repository root's, `AST_GREP_BIN` in the backend's, `LOOM_BROWSER_BIN` in the frontend's.

Type checking:

```bash
cd backend && mypy . && basedpyright .
cd frontend && npm run check && npm run lint
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
