FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG BAZELISK_VERSION=v1.29.0
ARG TARGETARCH

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        curl \
        file \
        gdb \
        git \
        gnupg \
        jq \
        less \
        lldb \
        lsb-release \
        make \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        strace \
        sudo \
        unzip \
        xz-utils \
        zip \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
        amd64) bazelisk_arch="amd64" ;; \
        arm64) bazelisk_arch="arm64" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL \
        -o /usr/local/bin/bazelisk \
        "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${bazelisk_arch}"; \
    chmod 0755 /usr/local/bin/bazelisk; \
    ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel

USER vscode
