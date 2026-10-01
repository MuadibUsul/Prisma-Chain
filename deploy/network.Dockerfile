FROM golang:1.24 AS chain-cli
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    cd chain && CGO_ENABLED=0 go build -trimpath -o /out/prismad ./cmd/prismad

FROM python:3.11-slim

WORKDIR /app
COPY network /app/network
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install '/app/network[chain,model]'
COPY --from=chain-cli /out/prismad /usr/local/bin/prismad

EXPOSE 8000 8080 8081
