# Multi-stage: build the static bundle with Node, check it with the deploy guard, then serve it
# with unprivileged nginx on port 8080.
#
# Base images are pinned by tag and by digest. The digest is the multi-arch index digest, so it
# resolves on amd64 and arm64 alike; Docker uses it and ignores the tag, which stays for readers.
# To refresh a pin: `docker buildx imagetools inspect <image>:<tag>` and copy the top "Digest:" line.
FROM node:22.23.1-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
# ADR 0001 D2: the image build fails if real market data could ship (the real-data marker anywhere
# in the app, a symlink, a JSON data file that is not synthetic, or public/data/real/). Runs after the
# build, so it also sees what is served. The marker's exact spelling is in scripts/guardDist.mjs.
RUN npm run guard:dist

# nginx-unprivileged rather than nginx plus a USER line: the nginx team ships it already set up to
# run as a non-root user (pid file, cache and temp dirs writable by that user, port 8080), so
# nothing here has to chown directories or move the pid file.
FROM nginxinc/nginx-unprivileged:1.28-alpine@sha256:7377697a821c131a924a7105fafbe7414db4e9fcc77a6f08f776f33f141ec3f8
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
# The base image already runs as uid 101 (nginx); stated here so it is visible and cannot drift.
USER 101
EXPOSE 8080
# 127.0.0.1, not localhost: in the container localhost resolves to ::1 first and nginx listens on
# IPv4 only, so a check against localhost always failed (measured on the previous nginx:1.27 image too).
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
