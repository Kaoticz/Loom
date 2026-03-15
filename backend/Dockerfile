FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    curl \
    && curl -L https://github.com/ast-grep/ast-grep/releases/download/v0.35.0/ast-grep-x86_64-unknown-linux-musl.tar.gz | tar -xz -C /usr/local/bin \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir .

CMD ["fastapi", "run", "main.py", "--port", "8000"]
