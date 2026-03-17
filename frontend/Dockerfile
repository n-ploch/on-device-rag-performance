# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# Point the API proxy at the orchestrator service name inside docker-compose
# (overridden at build time via --build-arg API_URL=...)
ARG API_URL=http://orchestrator-api:8080
ENV VITE_API_URL=${API_URL}
RUN npm run build

# ── Stage 2: serve ────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html

# Nginx config: serve SPA + proxy /api to the orchestrator API
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
