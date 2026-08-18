# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    export npm_config_fetch_retries=8 \
           npm_config_fetch_retry_mintimeout=20000 \
           npm_config_fetch_retry_maxtimeout=120000 && \
    attempt=1; \
    while [ "$attempt" -le 5 ]; do \
      npm ci --no-audit --no-fund && exit 0; \
      echo "npm ci failed on attempt $attempt; retrying with the shared cache"; \
      attempt=$((attempt + 1)); \
    done; \
    exit 1
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=6 CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
