FROM python:3.11-slim

WORKDIR /app
COPY network /app/network
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install /app/network

EXPOSE 8000 8080 8081
