"""Re-provision the canary trading wallets, WITHOUT ever printing a private key.

One reusable command for the whole wallet-rotation refresh we used to do by hand:

  1. mint N fresh trading wallets + a fresh treasury,
  2. sweep every current trading wallet into the new treasury,
  3. sweep the current treasury into the new treasury,
  4. top the new treasury up from the reserve (the canary wallet) if it is short,
  5. fund each new wallet to the requested SOL.

Keys are generated once, written to files on /data (0600) and recorded in the keyvault; nothing secret is ever
printed -- only addresses and the plan. Dry run by default; `--apply` moves the money. The new keys are staged to
`<file>.next` first, so a dry run shows the exact addresses an apply will use, and the live wallet file is only
swapped at the very end (the bot picks the new wallets up on its next restart).

  python -m tradingbot provision                         # interactive dry run (asks how many wallets + SOL each)
  python -m tradingbot provision --wallets 10 --sol 0.2  # dry run with the numbers given
  python -m tradingbot provision --wallets 10 --sol 0.2 --apply

Sources (read before anything is overwritten):
  current trading wallets : TB_LAUNCH_CANARY_WALLETS_PATH file (default /data/canary_wallets_new.json) else TB_CANARY_WALLETS
  current treasury        : /data/new_treasury.json else TB_TREASURY_KEY / OLD_TREASURY_KEY
  reserve (top-up source) : --reserve file, else TB_KEYPAIR_JSON / /run/secrets/keypair.json (the canary wallet)
"""
from __future__ import annotations

import json
import os
import sys
import time

LAMPORTS = 1_000_000_000
BASE_FEE = 5_000
FUND_BUFFER_SOL = 0.01    # left in the new treasury after funding
RESERVE_KEEP_SOL = 0.02   # left in the reserve after a top-up (never drained to 0)
TREASURY_NAME = "new_treasury.json"
WALLETS_NAME = "canary_wallets_new.json"
CANARY_SECRET = "/run/secrets/keypair.json"


def _data_dir() -> str:
    d = os.path.dirname(os.environ.get("TB_DB_PATH", "/data/tradingbot.db")) or "."
    return d if os.path.isdir(d) else "."


def _load_kp(text: str):
    from solders.keypair import Keypair
    text = text.strip().strip('"').strip("'").strip()
    if text.startswith("["):
        return Keypair.from_bytes(bytes(json.loads(text)))
    return Keypair.from_base58_string(text)


def _load_wallets(text: str) -> list:
    text = text.strip()
    out = []
    if text.startswith("["):
        for el in json.loads(text):
            out.append(_load_kp(json.dumps(el)) if isinstance(el, list) else _load_kp(str(el)))
    else:
        for part in text.replace(",", "\n").split("\n"):
            if part.strip():
                out.append(_load_kp(part))
    return out


def _write_secret(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(data))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _rpc_url() -> str:
    key = os.environ.get("HELIUS_API_KEY") or os.environ.get("BACKUP_HELIUS_API_KEY")
    return f"https://mainnet.helius-rpc.com/?api-key={key}" if key else "https://api.mainnet-beta.solana.com"


def _default_rpc(method: str, params: list):
    import urllib.request
    url = _rpc_url()
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    last = None
    for i in range(6):
        try:
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=30))
            if "result" in r:
                return r["result"]
            last = r.get("error")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(0.6 * (i + 1))
    print(f"  ! rpc {method} failed: {str(last)[:140]}")
    return None


class Provisioner:
    def __init__(self, apply: bool = False, rpc=None, reserve=None, reserve_keep: float = RESERVE_KEEP_SOL):
        self.apply = apply
        self.rpc = rpc or _default_rpc
        self.reserve_keep = reserve_keep
        self.reserve_path = reserve
        self.data = _data_dir()
        self.treasury_file = os.path.join(self.data, TREASURY_NAME)
        self.wallets_file = os.environ.get("TB_LAUNCH_CANARY_WALLETS_PATH") or os.path.join(self.data, WALLETS_NAME)

    # -- chain helpers --
    def balance(self, pub) -> int:
        r = self.rpc("getBalance", [str(pub)])
        return (r or {}).get("value", 0) if isinstance(r, dict) else 0

    def send(self, from_kp, to_pub, lamports: int, label: str) -> bool:
        from solders.hash import Hash
        from solders.message import MessageV0
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer
        from solders.transaction import VersionedTransaction
        to_pub = to_pub if isinstance(to_pub, Pubkey) else Pubkey.from_string(str(to_pub))
        print(f"  {label}: {str(from_kp.pubkey())[:6]} -> {str(to_pub)[:6]}  {lamports / LAMPORTS:.4f} SOL", flush=True)
        if lamports <= 0:
            print("    (nothing to send)")
            return True
        if not self.apply:
            return True
        ix = transfer(TransferParams(from_pubkey=from_kp.pubkey(), to_pubkey=to_pub, lamports=lamports))
        bh = (self.rpc("getLatestBlockhash", [{"commitment": "confirmed"}]) or {}).get("value", {}).get("blockhash")
        if not bh:
            print("    ! no blockhash"); return False
        import base64
        msg = MessageV0.try_compile(from_kp.pubkey(), [ix], [], Hash.from_string(bh))
        raw = base64.b64encode(bytes(VersionedTransaction(msg, [from_kp]))).decode()
        sig = self.rpc("sendTransaction", [raw, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}])
        if not sig:
            print("    ! send failed"); return False
        deadline = time.time() + 60
        while time.time() < deadline:
            st = (self.rpc("getSignatureStatuses", [[sig]]) or {}).get("value", [None])[0]
            if st and (st.get("err") or st.get("confirmationStatus") in ("confirmed", "finalized")):
                if st.get("err"):
                    print(f"    ! reverted: {st['err']}"); return False
                print(f"    ok {sig}"); return True
            time.sleep(1)
        print(f"    ! not confirmed: {sig}"); return False

    def sweep(self, kp, to_pub, label: str) -> int:
        bal = self.balance(kp.pubkey())
        amt = max(0, bal - BASE_FEE)
        self.send(kp, to_pub, amt, f"sweep {label} (bal {bal / LAMPORTS:.4f})")
        return amt

    # -- sources / new keys --
    def _read_old_wallets(self) -> list:
        if os.path.exists(self.wallets_file):
            with open(self.wallets_file) as fh:
                return _load_wallets(fh.read())
        env = os.environ.get("TB_CANARY_WALLETS")
        return _load_wallets(env) if env else []

    def _read_old_treasury(self):
        if os.path.exists(self.treasury_file):
            with open(self.treasury_file) as fh:
                return _load_kp(fh.read())
        env = os.environ.get("TB_TREASURY_KEY") or os.environ.get("OLD_TREASURY_KEY")
        return _load_kp(env) if env else None

    def _read_reserve(self):
        if self.reserve_path and os.path.exists(self.reserve_path):
            with open(self.reserve_path) as fh:
                return _load_kp(fh.read())
        ck = os.environ.get("TB_KEYPAIR_JSON")
        if not ck and os.path.exists(CANARY_SECRET):
            with open(CANARY_SECRET) as fh:
                ck = fh.read()
        return _load_kp(ck) if ck else None

    def _new_keys(self, n: int):
        """Staged new treasury + wallets: load `<file>.next` if present (so dry run and apply use the same keys),
        else generate and stage them (0600). Returns (treasury, [wallets], created_now)."""
        from solders.keypair import Keypair
        ts, ws = self.treasury_file + ".next", self.wallets_file + ".next"
        if os.path.exists(ts) and os.path.exists(ws):
            with open(ts) as fh:
                treasury = _load_kp(fh.read())
            with open(ws) as fh:
                wallets = _load_wallets(fh.read())
            if len(wallets) == n:
                return treasury, wallets, False
        treasury = Keypair()
        wallets = [Keypair() for _ in range(n)]
        _write_secret(ts, list(bytes(treasury)))
        _write_secret(ws, [list(bytes(k)) for k in wallets])
        return treasury, wallets, True

    def _vault(self, label: str, kp, note: str = "") -> None:
        try:
            from tradingbot import keyvault
            keyvault.record(label, kp, note)
        except Exception:  # noqa: BLE001
            pass

    def run(self, n: int, sol_each: float) -> None:
        old_wallets = self._read_old_wallets()
        old_treasury = self._read_old_treasury()
        reserve = self._read_reserve()
        treasury, wallets, created = self._new_keys(n)
        target = int(sol_each * LAMPORTS)
        need = n * target + (n + 2) * BASE_FEE + int(FUND_BUFFER_SOL * LAMPORTS)

        mode = "APPLY" if self.apply else "DRY RUN"
        print(f"=== provision ({mode}) === RPC {'helius' if (os.environ.get('HELIUS_API_KEY') or os.environ.get('BACKUP_HELIUS_API_KEY')) else 'public'}")
        print(f"mint {n} new wallets, fund {sol_each:.4f} SOL each (need ~{need / LAMPORTS:.4f} SOL in the new treasury)")
        print(f"staged key files ({'created now' if created else 'reused from a prior dry run'}; keys NOT printed):")
        print(f"  {self.treasury_file}.next   {self.wallets_file}.next")
        print(f"new treasury: {treasury.pubkey()}")
        print("new wallets:")
        for k in wallets:
            print("  ", k.pubkey())
        print(f"sources: {len(old_wallets)} current trading wallets, "
              f"{'current treasury ' + str(old_treasury.pubkey())[:6] if old_treasury else 'no current treasury'}, "
              f"{'reserve ' + str(reserve.pubkey())[:6] if reserve else 'NO reserve key'}")

        if not self.apply:
            print("\n(DRY RUN -- staged the new keys, moved no SOL. Re-run with --apply to execute.)")
            return

        self._vault("treasury-current", treasury)
        for i, k in enumerate(wallets):
            self._vault(f"wallet-{i}", k)

        print("\n-- 1. sweep current trading wallets -> new treasury --")
        for kp in old_wallets:
            self._vault(f"wallet-old-{str(kp.pubkey())[:6]}", kp, "superseded")
            self.sweep(kp, treasury.pubkey(), str(kp.pubkey())[:6])
        print("-- 2. sweep current treasury -> new treasury --")
        if old_treasury:
            self._vault("treasury-old", old_treasury, "superseded")
            self.sweep(old_treasury, treasury.pubkey(), "old-treasury")
        else:
            print("  (no current treasury to sweep)")

        tb = self.balance(treasury.pubkey())
        print(f"\n-- 3. top new treasury from the reserve if short (need {need / LAMPORTS:.4f}, have {tb / LAMPORTS:.4f}) --")
        if tb < need:
            if not reserve:
                print("  ! no reserve key available; cannot top up")
            else:
                rb = self.balance(reserve.pubkey())
                top = min(need - tb, rb - int(self.reserve_keep * LAMPORTS) - BASE_FEE)
                print(f"  reserve {str(reserve.pubkey())[:6]} balance {rb / LAMPORTS:.4f}")
                if top <= 0:
                    print(f"  ! reserve too low to top up (keeps {self.reserve_keep:.4f} SOL)")
                else:
                    self.send(reserve, treasury.pubkey(), top, "top reserve->treasury")

        print(f"\n-- 4. fund {n} new wallets to {sol_each:.4f} SOL each (top-to, so a re-run is safe) --")
        for kp in wallets:
            cur = self.balance(kp.pubkey())
            gap = target - cur
            if gap <= 0:
                print(f"  skip {str(kp.pubkey())[:6]} (already {cur / LAMPORTS:.4f})")
                continue
            self.send(treasury, kp.pubkey(), gap, f"fund {str(kp.pubkey())[:6]} ({cur / LAMPORTS:.4f} -> {sol_each:.4f})")

        # Swap the staged keys into the live treasury + wallet json on /data (the bot reads the wallet json via
        # TB_LAUNCH_CANARY_WALLETS_PATH, so this rotates the live set), then rebuild the keyvault + address book.
        os.replace(self.treasury_file + ".next", self.treasury_file)
        os.replace(self.wallets_file + ".next", self.wallets_file)
        try:
            from tradingbot import keyvault
            keyvault.cmd_sync()
        except Exception:  # noqa: BLE001
            pass
        print(f"\ntreasury final: {self.balance(treasury.pubkey()) / LAMPORTS:.4f} SOL")
        print("DONE. The live treasury + wallet json on /data are rotated (keys not shown).")
        print("Now rotate the Fly-secret backup too, so the new keys survive a volume wipe. Run on YOUR Mac")
        print("(the keys go /data -> Mac -> Fly via $(cat); they are never printed, and this restarts the bot")
        print("onto the new wallets):")
        print(f"  fly ssh sftp get {self.treasury_file} /tmp/t.json -a supermeme")
        print(f"  fly ssh sftp get {self.wallets_file} /tmp/w.json -a supermeme")
        print("  fly secrets set TB_TREASURY_KEY=\"$(cat /tmp/t.json)\" TB_CANARY_WALLETS=\"$(cat /tmp/w.json)\" "
              "-a supermeme && rm /tmp/t.json /tmp/w.json")


def _ask(prompt: str, cast):
    while True:
        try:
            return cast(input(prompt).strip())
        except (ValueError, EOFError):
            print("  please enter a number")


def main(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="provision", description="Re-provision the canary trading wallets.")
    p.add_argument("--wallets", type=int, help="how many new wallets to mint")
    p.add_argument("--sol", type=float, help="SOL to fund each new wallet with")
    p.add_argument("--reserve", help="reserve key file to top up from (default: the canary wallet)")
    p.add_argument("--reserve-keep", type=float, default=RESERVE_KEEP_SOL, help="SOL to leave in the reserve")
    p.add_argument("--apply", action="store_true", help="move the money (default: dry run)")
    a = p.parse_args(args)

    n, sol = a.wallets, a.sol
    if (n is None or sol is None) and sys.stdin.isatty():
        if n is None:
            n = _ask("How many new wallets to mint? ", int)
        if sol is None:
            sol = _ask("How much SOL to fund each wallet with? ", float)
    if n is None or sol is None:
        p.error("pass --wallets N and --sol X (or run on a terminal to be asked)")
    if n < 1 or sol <= 0:
        p.error("--wallets must be >= 1 and --sol > 0")
    Provisioner(apply=a.apply, reserve=a.reserve, reserve_keep=a.reserve_keep).run(n, sol)
