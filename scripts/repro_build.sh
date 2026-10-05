#!/bin/sh
# Reproducible prismad release build (roadmap B1-01c..e).
#
#   sh scripts/repro_build.sh [os] [arch]
#
# Builds the chain node twice from the same committed revision with fixed
# flags and asserts the two SHA256 digests are identical. Refuses to run
# on a dirty worktree: a release hash must name a committed revision.
#
#   GO=<go binary>  optional, default "go" (must be >= 1.23.2 for chain/)
#   OUT_DIR=<dir>   optional: copy one binary + prismad.sha256 + build-info.txt
#
# No wall-clock timestamp enters the binary: -buildvcs=false disables Go's
# VCS stamping (which would depend on worktree state) and no date is
# embedded by this project; the git commit is the timestamp authority.

set -eu

GO=${GO:-go}
OS=${1:-linux}
ARCH=${2:-amd64}

repo=$(git rev-parse --show-toplevel)
cd "$repo"

if [ -n "$(git status --porcelain)" ]; then
  echo "repro_build: worktree is dirty; commit first (release builds are reproducible only from a committed revision)" >&2
  exit 1
fi

git_sha=$(git rev-parse HEAD)
version=$(git describe --tags --match 'v*' --always)

out1=$(mktemp -d)
out2=$(mktemp -d)
trap 'rm -rf "$out1" "$out2"' EXIT

build() {
  out=$1
  (
    cd chain
    GOTOOLCHAIN=local "$GO" run ./tools/genprotocolversion
    # The committed generated file must equal the freeze artifact: drift
    # here would mean the embedded protocol identity is hand-edited.
    git diff --exit-code -- version/protocol_gen.go >/dev/null
    CGO_ENABLED=0 GOOS="$OS" GOARCH="$ARCH" GOTOOLCHAIN=local "$GO" build \
      -trimpath -buildvcs=false \
      -ldflags "-X prismachain/chain/version.gitCommit=$git_sha -X prismachain/chain/version.softwareVersion=$version" \
      -o "$out/prismad" ./cmd/prismad
  )
}

echo "repro_build: target $OS/$ARCH"
echo "  version : $version"
echo "  git sha : $git_sha"
echo "  go      : $(GOTOOLCHAIN=local "$GO" version)"

build "$out1"
build "$out2"

h1=$(sha256sum "$out1/prismad" | cut -d' ' -f1)
h2=$(sha256sum "$out2/prismad" | cut -d' ' -f1)
echo "  build 1 : $h1"
echo "  build 2 : $h2"

if [ "$h1" != "$h2" ]; then
  echo "repro_build: FAILED — the two builds differ; fix the build, do not document around it" >&2
  exit 1
fi
echo "  result  : IDENTICAL"

if [ -n "${OUT_DIR:-}" ]; then
  mkdir -p "$OUT_DIR"
  cp "$out1/prismad" "$OUT_DIR/prismad"
  printf '%s  prismad\n' "$h1" > "$OUT_DIR/prismad.sha256"
  {
    echo "software_version: $version"
    echo "git_commit: $git_sha"
    echo "build_target: $OS/$ARCH"
    echo "go: $(GOTOOLCHAIN=local "$GO" version)"
    echo "flags: CGO_ENABLED=0 -trimpath -buildvcs=false (no build date embedded)"
    echo "sha256: $h1"
  } > "$OUT_DIR/build-info.txt"
  echo "  artifacts: $OUT_DIR/prismad, $OUT_DIR/prismad.sha256, $OUT_DIR/build-info.txt"
fi
