FROM node:25-slim

# The node user is already created in the node image with UID 1000
WORKDIR /app
RUN chown node:node /app

USER node

COPY --chown=node:node package.json package-lock.json* ./
RUN npm install

COPY --chown=node:node . .

CMD ["npm", "run", "dev", "--", "--host"]
