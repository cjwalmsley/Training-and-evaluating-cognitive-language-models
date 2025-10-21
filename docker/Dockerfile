# syntax=docker/dockerfile:1

# Stage 1: Define the 'base' stage with common dependencies
FROM ubuntu:latest as base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Add packages needed by both build and final stages here
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Define the 'build' stage to compile the application
FROM base as build

ARG ANNABELL_VERSION="annabell"

RUN echo "Setting up build dependencies..." && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        cmake \
        libpcre3-dev \
        libgtest-dev \
        make \
        pkg-config \
        xz-utils && \
    cd /usr/src/gtest && \
    cmake CMakeLists.txt && \
    make && \
    rm -rf /var/lib/apt/lists/*

# Use a relative path for the source code from the build context
COPY ../source_code/${ANNABELL_VERSION}.tar.xz /app/

RUN echo "Setting up Annabell..." && \
    cd /app/ && \
    tar -xvJf ${ANNABELL_VERSION}.tar.xz && \
    cd ${ANNABELL_VERSION} && \
    ./configure && \
    make && \
    make install

# Stage 3: Define the 'final' production stage
FROM base AS final

WORKDIR /app

# Add sed for fixing line endings and other runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        adduser \
        libpcre3 \
        libgomp1 \
        sed && \
    rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the 'build' stage
COPY --from=build /usr/local/bin/annabell /usr/local/bin/annabell

# Use correct relative paths from the Dockerfile's context
COPY scripts/train_annabell_squad.sh .
COPY scripts/pre_train_annabell_squad_nyc.sh .
COPY scripts/test_annabell_squad.sh .

# Fix CRLF line endings and set execute permissions
RUN sed -i 's/\r$//' train_annabell_squad.sh && chmod +x train_annabell_squad.sh
RUN sed -i 's/\r$//' pre_train_annabell_squad_nyc.sh && chmod +x pre_train_annabell_squad_nyc.sh
RUN sed -i 's/\r$//' test_annabell_squad.sh && chmod +x test_annabell_squad.sh

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser