#!/bin/sh
# Container entrypoint.
# If a wallet key is supplied as a secret (TB_KEYPAIR_JSON), write it to an in-memory
# path readable only by this process. It is never written to the data volume, so it
# never ends up in volume snapshots or backups.
set -e

if [ -n "$TB_KEYPAIR_JSON" ]; then
    umask 077
    mkdir -p /run/secrets
    printf '%s' "$TB_KEYPAIR_JSON" > /run/secrets/keypair.json
    export TB_KEYPAIR_PATH=/run/secrets/keypair.json
    unset TB_KEYPAIR_JSON
fi

exec "$@"
