FROM golang:1.24 AS build
ARG GIT_SHA=unknown
ARG VERSION=dev
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod,sharing=locked \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    cd chain && CGO_ENABLED=0 go build -trimpath -buildvcs=false \
      -ldflags "-X prismachain/chain/version.gitCommit=${GIT_SHA} -X prismachain/chain/version.softwareVersion=${VERSION}" \
      -o /out/prismad ./cmd/prismad && \
    cd .. && CGO_ENABLED=0 go build -trimpath -buildvcs=false -o /out/prisma-vm ./cmd/prisma-vm

FROM debian:bookworm-slim
RUN apt-get update -qq && apt-get install -y -qq ca-certificates jq >/dev/null && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/prismad /usr/local/bin/prismad
COPY --from=build /out/prisma-vm /usr/local/bin/prisma-vm
COPY chain/local-entrypoint.sh /usr/local/bin/local-entrypoint
RUN chmod +x /usr/local/bin/local-entrypoint
EXPOSE 26656 26657 9090 1317
# Real endpoint check, bounded timeout, meaningful exit codes: the node's
# own RPC /status must answer within 3s. The image ships no secrets: the
# entrypoint generates the devnet test keyring at runtime in the data
# volume. -buildvcs=false + the explicit -ldflags keep the binary hash a
# pure function of the committed revision.
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
  CMD ["/usr/local/bin/prismad", "health"]
ENTRYPOINT ["/usr/local/bin/local-entrypoint"]
