FROM python:3.14-slim

RUN groupadd -g 1000 loom && \
    useradd -u 1000 -g loom -m loom

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    curl \
    unzip \
    && curl -L -o /tmp/ast-grep.zip https://github.com/ast-grep/ast-grep/releases/download/0.35.0/app-x86_64-unknown-linux-gnu.zip \
    && unzip /tmp/ast-grep.zip -d /usr/local/bin \
    && rm /tmp/ast-grep.zip \
    && rm -rf /var/lib/apt/lists/*

# Fix permissions
RUN chown -R loom:loom /app

USER loom
ENV PATH="/home/loom/.local/bin:${PATH}"

COPY --chown=loom:loom . .
RUN pip install --user --no-cache-dir .

CMD ["fastapi", "run", "main.py", "--port", "8000"]
