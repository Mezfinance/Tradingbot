#!/bin/sh
# Container entrypoint.
#
# 1. Wallet key: if supplied as a secret (TB_KEYPAIR_JSON), write it to an in-memory
#    path readable only by this process. It never touches the data volume, so it never
#    ends up in volume snapshots or backups.
# 2. Self-written code: if the bot has shipped its own code version (see
#    tradingbot/learning/codebase.py), run it instead of the image code, provided its
#    checksums verify, it was built for this image, and it is not crash-looping. On any
#    failure fall back to that version's parent, down to the image code. This runs in
#    shell on purpose: broken Python cannot stop the fallback.
set -e

if [ -n "$TB_KEYPAIR_JSON" ]; then
    (umask 077 && mkdir -p /run/secrets && printf '%s' "$TB_KEYPAIR_JSON" > /run/secrets/keypair.json)
    export TB_KEYPAIR_PATH=/run/secrets/keypair.json
    unset TB_KEYPAIR_JSON
fi

TB_BASE_HASH="$(cat "${TB_BASE_HASH_FILE:-/app/BASE_HASH}" 2>/dev/null || true)"
export TB_BASE_HASH
CODE_DIR="${TB_CODE_DIR:-$(dirname "${TB_DB_PATH:-/data/tradingbot.db}")/code}"
MAX_BOOTS="${TB_MAX_CRASH_BOOTS:-3}"

log() { echo "[entrypoint] $*"; }

fall_back() {  # $1 = version to abandon, $2 = reason
    parent="$(cat "$CODE_DIR/versions/$1/PARENT" 2>/dev/null || true)"
    echo "$1" >> "$CODE_DIR/crashed"
    if [ -n "$parent" ]; then
        log "abandoning $1 ($2); falling back to $parent"
        echo "$parent" > "$CODE_DIR/active"
    else
        log "abandoning $1 ($2); falling back to the image code"
        rm -f "$CODE_DIR/active"
    fi
    echo 0 > "$CODE_DIR/boots"
}

i=0
while [ -f "$CODE_DIR/active" ] && [ "$i" -lt 20 ]; do
    i=$((i + 1))
    V="$(cat "$CODE_DIR/active")"
    DIR="$CODE_DIR/versions/$V"
    if [ -z "$V" ] || [ ! -d "$DIR/tradingbot" ]; then
        log "active version '$V' not found; running the image code"
        rm -f "$CODE_DIR/active"
        break
    fi
    if [ "$(cat "$DIR/BASE_HASH" 2>/dev/null)" != "$TB_BASE_HASH" ]; then
        log "$V was built for a different image; running the new image code"
        rm -f "$CODE_DIR/active"
        break
    fi
    if ! (cd "$DIR" && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1); then
        fall_back "$V" "checksum mismatch"
        continue
    fi
    BOOTS=$(( $(cat "$CODE_DIR/boots" 2>/dev/null || echo 0) + 1 ))
    if [ "$BOOTS" -gt "$MAX_BOOTS" ]; then
        fall_back "$V" "crashed $((BOOTS - 1)) times in a row"
        continue
    fi
    echo "$BOOTS" > "$CODE_DIR/boots"
    export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
    export TB_CODE_VERSION="$V"
    log "running self-written code $V (boot $BOOTS of $MAX_BOOTS before fallback)"
    break
done

exec "$@"
