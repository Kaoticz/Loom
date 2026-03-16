FROM python:3.14-slim

RUN groupadd -g 1000 loom && \
    useradd -u 1000 -g loom -m loom

# RUN usermod -aG docker loom

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
