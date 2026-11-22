# syntax=docker/dockerfile:1
# Multi-stage build for the Next.js frontend. `dev` hot-reloads against a bind
# mount; `prod` runs the lean standalone server (see next.config.ts).
FROM node:22-alpine AS base
ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app

# Install dependencies from the committed lockfile (cached layer).
FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# Dev target: full deps + source, hot-reload via `next dev`. Bind all interfaces
# so the dev server is reachable through the published port.
FROM base AS dev
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0"]

# Build the standalone production bundle.
FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Prod target: minimal standalone server, no full node_modules. Runs as the
# node image's built-in unprivileged `node` user; the standalone output is
# chowned to it because Next.js writes runtime caches next to the server.
FROM base AS prod
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
