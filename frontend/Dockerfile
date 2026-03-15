FROM node:24-slim

# Pre-create the cache directory and ensure node owns it
RUN mkdir -p /tmp/.vite && chown -R node:node /tmp/.vite

USER node

WORKDIR /home/node

COPY --chown=node:node package.json package-lock.json* ./
RUN npm install

COPY --chown=node:node . .

# Environment variable to help Vite find the cache if config isn't enough
ENV VITE_CACHE_DIR=/tmp/.vite

CMD ["npm", "run", "dev", "--", "--host"]
