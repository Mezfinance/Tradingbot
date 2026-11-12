import hashlib
import json
import time

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from tradingbot import dashboard as dash
from tradingbot import dashpasskey as pk
from tradingbot.config import Config
from webauthn.helpers import bytes_to_base64url


def _cfg(tmp_path):
    return Config(db_path=str(tmp_path / "j.db"))


def test_rp_id_and_origin_from_host(monkeypatch):
    monkeypatch.delenv("TB_DASH_RP_ID", raising=False)
    monkeypatch.delenv("TB_DASH_ORIGIN", raising=False)
    assert pk.rp_id("supermeme.fly.dev") == "supermeme.fly.dev"
    assert pk.origin("supermeme.fly.dev") == "https://supermeme.fly.dev"
    assert pk.rp_id("localhost:18810") == "localhost"
    assert pk.origin("localhost:18810") == "http://localhost:18810"
    monkeypatch.setenv("TB_DASH_ORIGIN", "https://custom.example")
    assert pk.origin("whatever") == "https://custom.example"


def test_challenge_cookie_round_trips_and_rejects_tampering(tmp_path):
    cfg = _cfg(tmp_path)
    chal = b"\x01\x02\x03\x04random-challenge-bytes"
    sealed = pk.seal_challenge(cfg, chal, "login")
    assert pk.open_challenge(cfg, sealed, "login") == chal
    assert pk.open_challenge(cfg, sealed, "reg") is None            # wrong kind
    assert pk.open_challenge(cfg, sealed + "x", "login") is None    # tampered mac
    assert pk.open_challenge(cfg, "garbage", "login") is None


def test_challenge_expires(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    sealed = pk.seal_challenge(cfg, b"abc", "login")
    later = time.time() + pk.CHALLENGE_MAX_AGE_S + 5
    monkeypatch.setattr(pk.time, "time", lambda: later)
    assert pk.open_challenge(cfg, sealed, "login") is None


def test_session_cookie_valid_and_tamper_and_expiry(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    sess = pk.issue_session(cfg)
    assert pk.valid_session(cfg, sess)
    assert not pk.valid_session(cfg, sess + "0")
    assert not pk.valid_session(cfg, "")
    ts, mac = sess.split(".")
    stale = f"{int(ts) - pk.SESSION_MAX_AGE_S - 10}.{mac}"
    assert not pk.valid_session(cfg, stale)  # mac won't match the old ts anyway


def test_cred_store_add_list_remove(tmp_path):
    cfg = _cfg(tmp_path)
    assert not pk.has_passkeys(cfg)
    pk.save_creds(cfg, [{"id": "AAA", "public_key": "x", "sign_count": 0, "label": "iPhone", "created": 1, "last_used": None}])
    assert pk.has_passkeys(cfg)
    lst = pk.list_passkeys(cfg)
    assert lst[0]["label"] == "iPhone" and "public_key" not in lst[0]
    assert pk.remove_passkey(cfg, "AAA") is True
    assert pk.remove_passkey(cfg, "AAA") is False
    assert not pk.has_passkeys(cfg)


def test_registration_options_sets_challenge_and_excludes(tmp_path, monkeypatch):
    monkeypatch.delenv("TB_DASH_RP_ID", raising=False)
    cfg = _cfg(tmp_path)
    opts_json, cookie = pk.registration_options(cfg, "localhost")
    opts = json.loads(opts_json)
    assert opts["rp"]["id"] == "localhost" and "challenge" in opts
    assert pk.open_challenge(cfg, cookie, "reg") is not None


def _cose_key(pub):
    nums = pub.public_numbers()
    x = nums.x.to_bytes(32, "big")
    y = nums.y.to_bytes(32, "big")
    return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: x, -3: y})


def _assertion(cfg, host, priv, cose, challenge):
    """Hand-build a valid ES256 WebAuthn assertion for a stored credential."""
    rp_hash = hashlib.sha256(pk.rp_id(host).encode()).digest()
    auth_data = rp_hash + b"\x05" + (1).to_bytes(4, "big")  # flags UP+UV, signCount 1
    client = {"type": "webauthn.get", "challenge": bytes_to_base64url(challenge), "origin": pk.origin(host)}
    client_json = json.dumps(client).encode()
    client_hash = hashlib.sha256(client_json).digest()
    der = priv.sign(auth_data + client_hash, ec.ECDSA(hashes.SHA256()))
    return {
        "id": bytes_to_base64url(b"cred-1"),
        "rawId": bytes_to_base64url(b"cred-1"),
        "type": "public-key",
        "response": {
            "clientDataJSON": bytes_to_base64url(client_json),
            "authenticatorData": bytes_to_base64url(auth_data),
            "signature": bytes_to_base64url(der),
            "userHandle": None,
        },
        "clientExtensionResults": {},
    }


def test_authentication_verify_round_trip_and_bad_credential(tmp_path, monkeypatch):
    monkeypatch.delenv("TB_DASH_RP_ID", raising=False)
    monkeypatch.delenv("TB_DASH_ORIGIN", raising=False)
    cfg = _cfg(tmp_path)
    host = "localhost"
    priv = ec.generate_private_key(ec.SECP256R1())
    cid = bytes_to_base64url(b"cred-1")
    pk.save_creds(cfg, [{"id": cid, "public_key": bytes_to_base64url(_cose_key(priv.public_key())),
                         "sign_count": 0, "label": "test", "created": 1, "last_used": None}])
    challenge = b"the-login-challenge-value-123456"
    assertion = _assertion(cfg, host, priv, None, challenge)
    sess = pk.authentication_verify(cfg, host, assertion, challenge)
    assert sess and pk.valid_session(cfg, sess)
    assert pk.load_creds(cfg)[0]["last_used"] is not None  # recorded

    # an id we don't hold -> None, not an exception
    other = dict(assertion, id=bytes_to_base64url(b"nope"))
    assert pk.authentication_verify(cfg, host, other, challenge) is None


def test_authorised_accepts_a_passkey_session(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    sess = pk.issue_session(cfg)
    headers = {"Cookie": f"{pk.SESS_COOKIE}={sess}"}
    assert dash.authorised(headers, {}, cfg) is True
    bad = {"Cookie": f"{pk.SESS_COOKIE}={sess}0"}
    assert dash.authorised(bad, {}, cfg) is False
    assert dash.authorised({}, {}, cfg) is False  # no token, no session


def test_http_login_flow_end_to_end(tmp_path, monkeypatch):
    """GET options (sets the challenge cookie), sign the assertion, POST verify (sets the
    session cookie), then reach an authed endpoint with that session."""
    import http.client

    from tradingbot import health
    from tradingbot.journal import Journal

    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("TB_DASH_RP_ID", raising=False)
    monkeypatch.delenv("TB_DASH_ORIGIN", raising=False)
    cfg = _cfg(tmp_path)
    cfg.health_port = 18782
    host = "127.0.0.1:18782"
    priv = ec.generate_private_key(ec.SECP256R1())
    cid = bytes_to_base64url(b"cred-1")
    pk.save_creds(cfg, [{"id": cid, "public_key": bytes_to_base64url(_cose_key(priv.public_key())),
                         "sign_count": 0, "label": "test", "created": 1, "last_used": None}])
    j = Journal(cfg.db_path)  # a real file so the dashboard's read-only poller can open it
    board = dash.Dashboard(j, cfg)
    board.refresh()
    server = health.start_health_server(j, cfg, board)

    def req(method, path, cookie=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", 18782, timeout=5)
        h = {"Cookie": cookie} if cookie else {}
        if body is not None:
            h["Content-Type"] = "application/json"
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()

    try:
        # dashboard is served (passkeys exist even with no token); unauthed root is the sign-in page
        assert req("GET", "/")[0] == 200
        status = json.loads(req("GET", "/api/auth/status")[2])
        assert status["passkeys"] is True and status["authed"] is False

        code, hdrs, body = req("GET", "/api/auth/login/options")
        assert code == 200
        chal_cookie = hdrs["Set-Cookie"].split(";")[0]  # tb_chal=...
        challenge = __import__("webauthn").helpers.base64url_to_bytes(json.loads(body)["challenge"])

        assertion = _assertion(cfg, host, priv, None, challenge)
        code, hdrs, body = req("POST", "/api/auth/login/verify", cookie=chal_cookie,
                               body=json.dumps({"credential": assertion}))
        assert code == 200, body
        sess_cookie = hdrs["Set-Cookie"].split(";")[0]  # tb_sess=...
        assert sess_cookie.startswith(pk.SESS_COOKIE + "=")

        # the session now reaches an authed endpoint, and register is gated to it
        assert req("GET", "/api/summary", cookie=sess_cookie)[0] == 200
        assert req("GET", "/api/auth/register/options", cookie=None)[0] == 401
        assert req("GET", "/api/auth/register/options", cookie=sess_cookie)[0] == 200

        # logout clears the session
        code, hdrs, _ = req("POST", "/api/auth/logout", cookie=sess_cookie)
        assert code == 200 and f"{pk.SESS_COOKIE}=;" in hdrs["Set-Cookie"]
    finally:
        server.shutdown()
