# Development image. It runs the Vite dev server so that edits to the bind
# mounted source reload immediately. It is not suitable for production: Vite's
# dev server is unoptimised and binds openly. A production image would add a
# build stage running `npm run build` and serve the output with adapter-node.
FROM node:24-slim

# WORKDIR must match the path docker-compose bind mounts the source to. When they
# differ, the container runs the copy baked into the image and edits on the host
# have no effect, which looks like hot reload being broken.
WORKDIR /app

# Vite writes its cache here; node must own it before dropping privileges.
RUN mkdir -p /tmp/.vite /app && chown -R node:node /tmp/.vite /app

USER node

COPY --chown=node:node package.json package-lock.json* ./
RUN npm install

COPY --chown=node:node . .

ENV VITE_CACHE_DIR=/tmp/.vite

CMD ["npm", "run", "dev", "--", "--host"]
