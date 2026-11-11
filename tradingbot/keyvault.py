"""Labelled recovery vault for every keypair in the canary flow.

Records each keypair (label, address, created-at, notes, secret) next to the journal so the whole set
can always be recovered. Two files are kept in the DB directory (``/data`` on Fly):

  keyvault.json    (0600) the vault WITH secrets. Each entry's "secret" is the 64-byte keypair int-array.
  addressbook.json (0644) a key-free replica: every treasury/wallet address ever, with labels -- no secrets.
                   Rewritten on every change; this is what the open /api/addressbook endpoint serves, so the
                   web server never opens the file that holds keys.

CLI (``python -m tradingbot keyvault <cmd>``):
  list                      label | address | created  (NO secrets)
  import                    record the current keys (canary, treasury, wallets) from env + /data files
  sync                      rebuild the key-free address book from the vault
  add <label> <src>         record a keypair from a file path, env-var name, or inline key text
  note <address> <text...>  annotate an entry (e.g. mark a lost key)
  export <outpath>          copy the full vault (WITH secrets) to a path for download

The entrypoint runs ``keyvault import`` on boot when ``TB_KEYVAULT=1``, so both files stay current with no
manual step.
"""
from __future__ import annotations

import json
import os
import time


def _dir() -> str:
    d = os.path.dirname(os.environ.get("TB_DB_PATH", "/data/tradingbot.db")) or "."
    return d if os.path.isdir(d) else "."


def _vault_path() -> str:
    return os.path.join(_dir(), "keyvault.json")


def _public_path() -> str:
    return os.path.join(_dir(), "addressbook.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _kp(text: str):
    from solders.keypair import Keypair

    text = text.strip().strip('"').strip("'").strip()
    if text.startswith("["):
        return Keypair.from_bytes(bytes(json.loads(text)))
    return Keypair.from_base58_string(text)


def _wallets(text: str) -> list:
    from solders.keypair import Keypair

    text = text.strip()
    out = []
    if text.startswith("["):
        data = json.loads(text)
        if data and isinstance(data[0], int):            # a single int-array keypair
            out.append(Keypair.from_bytes(bytes(data)))
        else:                                            # an array of keypairs
            for el in data:
                out.append(Keypair.from_bytes(bytes(el)) if isinstance(el, list) else _kp(str(el)))
    else:
        for p in text.replace(",", "\n").split("\n"):
            if p.strip():
                out.append(_kp(p))
    return out


def _load() -> dict:
    path = _vault_path()
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"version": 1, "keys": {}}


def _public(v: dict) -> dict:
    """A key-free view of the vault: address -> {label, created, notes}. No secret ever."""
    return {
        "version": 1,
        "updated": _now(),
        "count": len(v["keys"]),
        "addresses": {
            a: {"label": e.get("label"), "created": e.get("created"), "notes": e.get("notes", [])}
            for a, e in v["keys"].items()
        },
    }


def _save(v: dict) -> None:
    path = _vault_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(v, fh)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    # key-free replica, rewritten on every change (what the web server serves)
    pub, ppath = _public(v), _public_path()
    ptmp = ppath + ".tmp"
    with open(ptmp, "w") as fh:
        json.dump(pub, fh, indent=2)
    os.chmod(ptmp, 0o644)
    os.replace(ptmp, ppath)


def record(label: str, kp, note: str = "") -> str:
    """Add or update one keypair by address. Returns its address."""
    v = _load()
    addr = str(kp.pubkey())
    e = v["keys"].get(addr) or {"created": _now(), "notes": []}
    e["label"] = label
    e["secret"] = list(bytes(kp))
    if note and note not in e.get("notes", []):
        e.setdefault("notes", []).append(note)
    v["keys"][addr] = e
    _save(v)
    return addr


def _read_source(src: str) -> str:
    if os.path.exists(src):
        with open(src) as fh:
            return fh.read()
    if src in os.environ:
        return os.environ[src]
    return src                                           # treat as inline key text


def cmd_import() -> None:
    d = _dir()
    # canary wallet (funding source)
    ck = os.environ.get("TB_KEYPAIR_JSON")
    if not ck and os.path.exists("/run/secrets/keypair.json"):
        ck = open("/run/secrets/keypair.json").read()
    if ck:
        record("canary", _kp(ck))
    # current treasury (secret or /data file)
    tfile = os.path.join(d, "new_treasury.json")
    if os.environ.get("TB_TREASURY_KEY"):
        record("treasury-current", _kp(os.environ["TB_TREASURY_KEY"]))
    if os.path.exists(tfile):
        record("treasury-current", _kp(open(tfile).read()))
    # an old treasury, if one is passed in the env
    if os.environ.get("OLD_TREASURY_KEY"):
        record("treasury-old", _kp(os.environ["OLD_TREASURY_KEY"]), "superseded")
    # trading wallets: the /data file (current) and the TB_CANARY_WALLETS secret (may be the prior set)
    wfile = os.path.join(d, "canary_wallets_new.json")
    if os.path.exists(wfile):
        for i, kp in enumerate(_wallets(open(wfile).read())):
            record(f"wallet-{i}", kp)
    if os.environ.get("TB_CANARY_WALLETS"):
        for kp in _wallets(os.environ["TB_CANARY_WALLETS"]):     # idempotent by address
            if str(kp.pubkey()) not in _load()["keys"]:
                record(f"wallet-old-{str(kp.pubkey())[:6]}", kp, "superseded")
    v = _load()
    print(f"vault now holds {len(v['keys'])} keypair(s) at {_vault_path()}")
    cmd_list()


def cmd_list() -> None:
    v = _load()
    rows = sorted(v["keys"].items(), key=lambda kv: (kv[1].get("label", ""), kv[0]))
    for addr, e in rows:
        notes = ("  # " + "; ".join(e["notes"])) if e.get("notes") else ""
        has = "" if e.get("secret") else "  (ADDRESS ONLY -- key not held)"
        print(f"  {e.get('label', '?'):20} {addr}  {e.get('created', '')}{has}{notes}")
    print(f"total: {len(v['keys'])}  (secrets are stored in the file, not shown here)")


def cmd_sync() -> None:
    v = _load()
    _save(v)                      # rewrites both files from the current vault
    print(f"synced {len(v['keys'])} address(es) -> {_public_path()}")


def cmd_add(label: str, src: str) -> None:
    kps = _wallets(_read_source(src))
    for i, kp in enumerate(kps):
        lbl = label if len(kps) == 1 else f"{label}-{i}"
        print(f"recorded {lbl}: {record(lbl, kp)}")


def cmd_note(addr: str, text: str) -> None:
    v = _load()
    if addr not in v["keys"]:
        v["keys"][addr] = {"label": "(address only)", "created": _now(), "notes": [], "secret": None}
    v["keys"][addr].setdefault("notes", []).append(text)
    _save(v)
    print(f"noted {addr}")


def cmd_export(outpath: str) -> None:
    v = _load()
    with open(outpath, "w") as fh:
        json.dump(v, fh, indent=2)
    os.chmod(outpath, 0o600)
    print(f"exported {len(v['keys'])} keypair(s) to {outpath} (0600, CONTAINS SECRETS -- store safely)")


def main(args: list[str]) -> None:
    if not args or args[0] == "list":
        cmd_list()
    elif args[0] == "import":
        cmd_import()
    elif args[0] == "sync":
        cmd_sync()
    elif args[0] == "add" and len(args) >= 3:
        cmd_add(args[1], args[2])
    elif args[0] == "note" and len(args) >= 3:
        cmd_note(args[1], " ".join(args[2:]))
    elif args[0] == "export" and len(args) >= 2:
        cmd_export(args[1])
    else:
        print(__doc__)
