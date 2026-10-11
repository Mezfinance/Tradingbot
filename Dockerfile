ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TB_TESTS_DIR=/app/tests

WORKDIR /app
COPY pyproject.toml README.md ./
COPY tradingbot ./tradingbot
# [live] adds solders (canary/live trades); [backup] adds boto3 (daily uploads to object storage);
# [engineer] adds pytest, which the bot uses to test its own code changes before shipping them.
RUN pip install ".[live,backup,engineer]"

# The test suite ships with the image: self-written code must pass it before going live.
COPY tests ./tests

# Fingerprint of this image's code. Self-written versions record which image they were built
# on, and the entrypoint ignores versions built for a different (older) image.
RUN python -c "from tradingbot.learning.codebase import package_hash, running_package_dir; \
print(package_hash(running_package_dir()))" > /app/BASE_HASH

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "tradingbot", "run"]
