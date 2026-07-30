FROM python:3.14-slim

RUN groupadd -g 1000 loom && \
    useradd -u 1000 -g loom -m loom

# The backend spawns sandbox containers through the host's Docker socket, so the
# 'loom' user must belong to the group that owns it. GID 999 is the usual owner of
# /var/run/docker.sock on Debian-based hosts; override DOCKER_GID at build time to
# match your host if it differs (`stat -c '%g' /var/run/docker.sock`).
ARG DOCKER_GID=999
RUN groupadd -g ${DOCKER_GID} docker 2>/dev/null || groupmod -n docker $(getent group ${DOCKER_GID} | cut -d: -f1) && \
    usermod -aG docker loom

# Create a virtualenv and put it in the PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git docker.io curl unzip \
    && curl -L -o /tmp/ast-grep.zip https://github.com/ast-grep/ast-grep/releases/download/0.35.0/app-x86_64-unknown-linux-gnu.zip \
    && unzip /tmp/ast-grep.zip -d /usr/local/bin \
    && rm /tmp/ast-grep.zip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into the venv
COPY . .
RUN pip install --no-cache-dir .

# Ensure the 'loom' user owns the app and the venv
RUN chown -R loom:loom /app $VIRTUAL_ENV

USER loom
CMD ["fastapi", "run", "main.py", "--port", "8000"]
