# syntax=docker/dockerfile:1

FROM ubuntu:latest as base

FROM base as build

RUN echo "Setting up dependencies..." && \
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

COPY annabell_increased_phrase_size.tar.xz /app/

RUN echo "Setting up Annabell..." && \
    cd /app/ && \
    tar -xvJf annabell_increased_phrase_size.tar.xz && \
    echo "Contents of /app after tar:" && \
    ls -l /app/ && \
    # Assuming tar extracts to a directory named 'annabell' inside /app
    cd annabell_increased_phrase_size && \
    ./configure && \
    make && \
    make install

FROM base AS final

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        adduser \
        libpcre3 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/annabell /usr/local/bin/annabell
COPY train_annabell_squad.sh .
RUN chmod +x train_annabell_squad.sh

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

#ENTRYPOINT [ \
#    "./train_annabell_squad.sh", \
#    "data/statements/crossvalidation/round1/logs/logfile_squad_statements.txt", \
#    "data/statements/crossvalidation/round1/links/links_people_body_skills.dat", \
#    "data/statements/crossvalidation/round1/links/links_people_body_skills_squad.dat", \
#    "data/statements/training/all_statements.txt" \
#]

#ENTRYPOINT ["annabell"]