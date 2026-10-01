FROM golang:1.24 AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    cd chain && CGO_ENABLED=0 go build -trimpath -o /out/prismad ./cmd/prismad

FROM debian:bookworm-slim
RUN apt-get update -qq && apt-get install -y -qq ca-certificates jq >/dev/null && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/prismad /usr/local/bin/prismad
COPY chain/local-entrypoint.sh /usr/local/bin/local-entrypoint
RUN chmod +x /usr/local/bin/local-entrypoint
EXPOSE 26656 26657 9090 1317
ENTRYPOINT ["/usr/local/bin/local-entrypoint"]
