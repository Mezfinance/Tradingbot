"""Re-provisioning the canary wallets: sweep -> top up from the reserve -> fund, and rotate the live json files,
without ever printing a key. The chain is faked as a balance dict so the orchestration is tested, not signing/RPC."""
import json

import pytest

solders = pytest.importorskip("solders")
from solders.keypair import Keypair

from tradingbot import provision as pv
from tradingbot.provision import BASE_FEE, LAMPORTS, Provisioner

SOL = LAMPORTS


class FakeProv(Provisioner):
    """Provisioner with the chain replaced by an in-memory balance dict."""

    def __init__(self, balances, **kw):
        super().__init__(**kw)
        self._bal = {str(k): int(v) for k, v in balances.items()}
        self.sends = []

    def balance(self, pub):
        return self._bal.get(str(pub), 0)

    def send(self, from_kp, to_pub, lamports, label):
        if lamports <= 0:
            return True
        self.sends.append((str(from_kp.pubkey()), str(to_pub), lamports))
        if self.apply:
            self._bal[str(from_kp.pubkey())] = self._bal.get(str(from_kp.pubkey()), 0) - lamports - BASE_FEE
            self._bal[str(to_pub)] = self._bal.get(str(to_pub), 0) + lamports
        return True


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "tradingbot.db"))
    for k in ("TB_LAUNCH_CANARY_WALLETS_PATH", "TB_CANARY_WALLETS", "TB_TREASURY_KEY",
              "OLD_TREASURY_KEY", "TB_KEYPAIR_JSON", "HELIUS_API_KEY", "BACKUP_HELIUS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    old = [Keypair() for _ in range(3)]
    treasury = Keypair()
    reserve = Keypair()
    (tmp_path / "canary_wallets_new.json").write_text(json.dumps([list(bytes(k)) for k in old]))
    (tmp_path / "new_treasury.json").write_text(json.dumps(list(bytes(treasury))))
    monkeypatch.setenv("TB_KEYPAIR_JSON", json.dumps(list(bytes(reserve))))  # the canary wallet = the reserve
    return tmp_path, old, treasury, reserve


def test_dry_run_stages_keys_and_moves_nothing(env):
    tmp, old, treasury, reserve = env
    prov = FakeProv({k.pubkey(): SOL // 20 for k in old}, apply=False)
    prov.run(3, 0.2)
    assert prov.sends == []  # no transfers on a dry run
    assert (tmp / "new_treasury.json.next").exists() and (tmp / "canary_wallets_new.json.next").exists()
    live = json.loads((tmp / "canary_wallets_new.json").read_text())  # live file untouched: still the old 3
    assert [str(Keypair.from_bytes(bytes(e)).pubkey()) for e in live] == [str(k.pubkey()) for k in old]


def test_apply_sweeps_tops_up_from_reserve_funds_and_rotates(env):
    tmp, old, treasury, reserve = env
    bal = {k.pubkey(): SOL // 20 for k in old}  # 0.05 SOL each: short of 3 x 0.2, so the reserve must top up
    bal[reserve.pubkey()] = 2 * SOL
    prov = FakeProv(bal, apply=True)
    prov.run(3, 0.2)

    froms = [f for f, _, _ in prov.sends]
    assert all(str(k.pubkey()) in froms for k in old)       # every current wallet swept
    assert str(reserve.pubkey()) in froms                    # the reserve topped the treasury up

    new = json.loads((tmp / "canary_wallets_new.json").read_text())  # live file rotated to the 3 new wallets
    new_pubs = [str(Keypair.from_bytes(bytes(e)).pubkey()) for e in new]
    assert len(new_pubs) == 3 and not set(new_pubs) & {str(k.pubkey()) for k in old}
    for e in new:  # each new wallet funded to 0.2 SOL
        assert prov.balance(Keypair.from_bytes(bytes(e)).pubkey()) == int(0.2 * SOL)
    new_treasury = str(Keypair.from_bytes(bytes(json.loads((tmp / "new_treasury.json").read_text()))).pubkey())
    assert new_treasury != str(treasury.pubkey())            # treasury json rotated too
    assert not (tmp / "canary_wallets_new.json.next").exists()  # staging swapped into place

    book = json.loads((tmp / "addressbook.json").read_text())  # key-free address book updated
    assert new_treasury in book["addresses"] and "secret" not in json.dumps(book)


def test_requires_wallets_and_sol_when_not_a_tty(env, monkeypatch):
    monkeypatch.setattr(pv.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit):
        pv.main([])  # no --wallets/--sol and no terminal to ask
