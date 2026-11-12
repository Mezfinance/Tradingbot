"""Passkey (WebAuthn) sign-in for the dashboard.

The dashboard was gated by a single shared `DASHBOARD_TOKEN`. This adds per-device
passkeys (FaceID on a phone, TouchID / Windows Hello on a computer, or a hardware
security key), so the Fly app is account-gated by the owner's own devices rather than
by a password that has to be pasted. The token stays as the **break-glass bootstrap**:
you open the page once with `?key=<DASHBOARD_TOKEN>` to register the first passkey (or
to recover if every device is lost), and after that you sign in with a passkey.

Everything here is self-contained — no database, no external service. Registered
credentials (public keys only, never a private key — those never leave the device's
secure element) live in a small JSON file next to the journal. Sessions and the
ceremony challenges are stateless, carried in HMAC-signed cookies keyed by a random
server secret persisted on the data volume, so they survive a restart but a stolen
cookie can't be forged.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time

log = logging.getLogger(__name__)

PASSKEYS_FILE = "dash_passkeys.json"
SESSION_KEY_FILE = "dash_session_key"

CHAL_COOKIE = "tb_chal"   # short-lived signed cookie carrying the ceremony challenge
SESS_COOKIE = "tb_sess"   # signed session cookie set after a passkey sign-in

SESSION_MAX_AGE_S = 30 * 86400   # a passkey session lasts 30 days
CHALLENGE_MAX_AGE_S = 300        # a register/login ceremony must finish within 5 minutes

RP_NAME = "MZFq Quant Desk"
USER_ID = b"mzfq-owner"
USER_NAME = "owner"
USER_DISPLAY = "MZFq owner"

_KEY_LOCK = threading.Lock()


# --- file locations -------------------------------------------------------------------

def _data_dir(cfg) -> str:
    path = getattr(cfg, "db_path", "") or ""
    d = os.path.dirname(path)
    return d or "."


def passkeys_path(cfg) -> str:
    return os.path.join(_data_dir(cfg), PASSKEYS_FILE)


def _session_key(cfg) -> bytes:
    """A stable random secret for signing cookies. Env override wins; otherwise persisted
    on the data volume so sessions survive a restart but are unguessable."""
    env = os.environ.get("TB_DASH_SESSION_SECRET")
    if env:
        return env.encode()
    path = os.path.join(_data_dir(cfg), SESSION_KEY_FILE)
    with _KEY_LOCK:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw:
                return bytes.fromhex(raw)
        except (OSError, ValueError):
            pass
        key = secrets.token_bytes(32)
        try:
            with open(path, "w") as fh:
                fh.write(key.hex())
            os.chmod(path, 0o600)
        except OSError as e:  # noqa: BLE001
            log.warning("could not persist the dashboard session key: %s", e)
        return key


# --- relying-party identity (derived from the request host) ---------------------------

def rp_id(host: str | None) -> str:
    override = os.environ.get("TB_DASH_RP_ID")
    if override:
        return override
    return (host or "localhost").split(":")[0]


def origin(host: str | None) -> str:
    override = os.environ.get("TB_DASH_ORIGIN")
    if override:
        return override
    h = host or "localhost"
    scheme = "http" if h.split(":")[0] in ("localhost", "127.0.0.1") else "https"
    return f"{scheme}://{h}"


# --- credential store -----------------------------------------------------------------

def load_creds(cfg) -> list:
    try:
        with open(passkeys_path(cfg)) as fh:
            return json.load(fh) or []
    except (OSError, ValueError):
        return []


def save_creds(cfg, creds: list) -> None:
    path = passkeys_path(cfg)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(creds, fh, indent=1)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def has_passkeys(cfg) -> bool:
    return bool(load_creds(cfg))


def list_passkeys(cfg) -> list:
    """Public (non-secret) view for the management UI: credential id, label and dates."""
    out = []
    for c in load_creds(cfg):
        out.append({
            "id": c["id"],
            "label": c.get("label") or "passkey",
            "created": c.get("created"),
            "last_used": c.get("last_used"),
        })
    return out


def remove_passkey(cfg, cred_id: str) -> bool:
    creds = load_creds(cfg)
    kept = [c for c in creds if c["id"] != cred_id]
    if len(kept) == len(creds):
        return False
    save_creds(cfg, kept)
    return True


# --- signed cookies (sessions and ceremony challenges) --------------------------------

def _sign(cfg, msg: str) -> str:
    import hmac
    import hashlib
    return hmac.new(_session_key(cfg), msg.encode(), hashlib.sha256).hexdigest()


def seal_challenge(cfg, challenge: bytes, kind: str) -> str:
    from webauthn.helpers import bytes_to_base64url
    ts = str(int(time.time()))
    payload = f"{kind}.{bytes_to_base64url(challenge)}.{ts}"
    return f"{payload}.{_sign(cfg, payload)}"


def open_challenge(cfg, cookie_value: str, kind: str) -> bytes | None:
    import hmac
    from webauthn.helpers import base64url_to_bytes
    try:
        k, ch, ts, mac = (cookie_value or "").split(".")
    except ValueError:
        return None
    payload = f"{k}.{ch}.{ts}"
    if not hmac.compare_digest(mac, _sign(cfg, payload)):
        return None
    if k != kind:
        return None
    if time.time() - int(ts) > CHALLENGE_MAX_AGE_S:
        return None
    try:
        return base64url_to_bytes(ch)
    except Exception:  # noqa: BLE001
        return None


def issue_session(cfg) -> str:
    ts = str(int(time.time()))
    return f"{ts}.{_sign(cfg, 'sess.' + ts)}"


def valid_session(cfg, cookie_value: str) -> bool:
    import hmac
    try:
        ts, mac = (cookie_value or "").split(".")
    except (ValueError, AttributeError):
        return False
    if not hmac.compare_digest(mac, _sign(cfg, "sess." + ts)):
        return False
    try:
        return time.time() - int(ts) <= SESSION_MAX_AGE_S
    except ValueError:
        return False


# --- WebAuthn ceremonies --------------------------------------------------------------

def registration_options(cfg, host: str) -> tuple[str, str]:
    """Return (options JSON for navigator.credentials.create, signed challenge cookie)."""
    from webauthn import generate_registration_options, options_to_json
    from webauthn.helpers import base64url_to_bytes
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    )
    exclude = [PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in load_creds(cfg)]
    opts = generate_registration_options(
        rp_id=rp_id(host),
        rp_name=RP_NAME,
        user_name=USER_NAME,
        user_id=USER_ID,
        user_display_name=USER_DISPLAY,
        exclude_credentials=exclude or None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
    )
    return options_to_json(opts), seal_challenge(cfg, opts.challenge, "reg")


def register_verify(cfg, host: str, credential, challenge: bytes, label: str) -> dict:
    from webauthn import verify_registration_response
    from webauthn.helpers import bytes_to_base64url
    body = credential if isinstance(credential, str) else json.dumps(credential)
    v = verify_registration_response(
        credential=body,
        expected_challenge=challenge,
        expected_rp_id=rp_id(host),
        expected_origin=origin(host),
        require_user_verification=False,
    )
    cid = bytes_to_base64url(v.credential_id)
    creds = load_creds(cfg)
    if any(c["id"] == cid for c in creds):
        raise ValueError("that passkey is already registered")
    creds.append({
        "id": cid,
        "public_key": bytes_to_base64url(v.credential_public_key),
        "sign_count": v.sign_count,
        "label": (label or "passkey").strip()[:40] or "passkey",
        "created": int(time.time()),
        "last_used": None,
    })
    save_creds(cfg, creds)
    return {"ok": True, "passkeys": list_passkeys(cfg)}


def authentication_options(cfg, host: str) -> tuple[str, str]:
    """Return (options JSON for navigator.credentials.get, signed challenge cookie)."""
    from webauthn import generate_authentication_options, options_to_json
    from webauthn.helpers import base64url_to_bytes
    from webauthn.helpers.structs import PublicKeyCredentialDescriptor, UserVerificationRequirement
    allow = [PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in load_creds(cfg)]
    opts = generate_authentication_options(
        rp_id=rp_id(host),
        allow_credentials=allow or None,
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    return options_to_json(opts), seal_challenge(cfg, opts.challenge, "login")


def authentication_verify(cfg, host: str, credential, challenge: bytes) -> str | None:
    """Verify an assertion against a stored public key. Returns a session cookie value
    on success, or None if the credential is unknown or verification fails."""
    from webauthn import verify_authentication_response
    data = credential if isinstance(credential, dict) else json.loads(credential)
    cid = data.get("id") or data.get("rawId")
    creds = load_creds(cfg)
    cred = next((c for c in creds if c["id"] == cid), None)
    if cred is None:
        return None
    from webauthn.helpers import base64url_to_bytes
    v = verify_authentication_response(
        credential=json.dumps(data),
        expected_challenge=challenge,
        expected_rp_id=rp_id(host),
        expected_origin=origin(host),
        credential_public_key=base64url_to_bytes(cred["public_key"]),
        credential_current_sign_count=cred["sign_count"],
        require_user_verification=False,
    )
    cred["sign_count"] = v.new_sign_count
    cred["last_used"] = int(time.time())
    save_creds(cfg, creds)
    return issue_session(cfg)
