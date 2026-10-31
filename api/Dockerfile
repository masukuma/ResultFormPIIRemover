# rfpii-api, with the rfpii binary it runs. Build from the repository root:
#
#   mkdir -p dist && cp rust/target/release/rfpii dist/ && cp -a /opt dist/opt
#   docker build -f api/Dockerfile -t rfpii-api .
#
# The image needs rfpii and its native libraries: rfpii is built against
# OpenCV, MuPDF, ONNX Runtime (with the PP-OCR models) and OpenBLAS under
# /opt (rust/scripts/setup-deps.sh), and finds them there (RUNPATH). Copy
# that /opt into dist/opt, and the binary into dist/. The templates come
# from templates/. TLS is terminated by a reverse proxy in front (API.md,
# Deployment).

FROM rust:1-slim-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src/api
COPY api/ ./
RUN cargo build --release --locked

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libssl3t64 libstdc++6 tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --home /data --shell /usr/sbin/nologin rfpii
COPY dist/opt/ /opt/
COPY dist/rfpii /usr/local/bin/rfpii
COPY templates/ /opt/rfpii/templates/
COPY --from=build /src/api/target/release/rfpii-api /usr/local/bin/rfpii-api
ENV DATA_DIR=/data \
    BIND=0.0.0.0:8080 \
    RFPII_BIN=/usr/local/bin/rfpii \
    RFPII_TEMPLATES=/opt/rfpii/templates \
    SIGN_KEY=/keys/signing_key.pem \
    PLATFORM_COUNTRY=KE
RUN mkdir -p /data && chown rfpii /data
VOLUME ["/data"]
USER rfpii
EXPOSE 8080
# health: GET /healthz (alive) and GET /readyz (database, disk, rfpii), for
# the orchestrator or the proxy to probe
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/rfpii-api"]
CMD ["serve", "--workers", "2"]
