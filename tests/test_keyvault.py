import json
import os

import pytest

from tradingbot import keyvault


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "tradingbot.db"))  # vault + addressbook land here
    for k in ("TB_KEYPAIR_JSON", "TB_CANARY_WALLETS", "TB_TREASURY_KEY", "OLD_TREASURY_KEY"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_record_writes_vault_and_keyfree_replica(vault):
    from solders.keypair import Keypair

    a = keyvault.record("treasury-current", Keypair())
    b = keyvault.record("wallet-0", Keypair())
    held = json.load(open(vault / "keyvault.json"))
    pub = json.load(open(vault / "addressbook.json"))
    assert set(held["keys"]) == {a, b}
    assert all("secret" in e for e in held["keys"].values())           # vault keeps the keys
    assert set(pub["addresses"]) == {a, b} and pub["count"] == 2
    assert "secret" not in json.dumps(pub)                             # replica is key-free
    assert pub["addresses"][a]["label"] == "treasury-current"
    assert oct(os.stat(vault / "keyvault.json").st_mode)[-3:] == "600"


def test_import_captures_treasury_and_wallets(vault):
    from solders.keypair import Keypair

    treasury = Keypair()
    (vault / "new_treasury.json").write_text(json.dumps(list(bytes(treasury))))
    wallets = [Keypair() for _ in range(3)]
    (vault / "canary_wallets_new.json").write_text(json.dumps([list(bytes(k)) for k in wallets]))
    keyvault.cmd_import()
    pub = json.load(open(vault / "addressbook.json"))
    assert pub["addresses"][str(treasury.pubkey())]["label"] == "treasury-current"
    for i, k in enumerate(wallets):
        assert pub["addresses"][str(k.pubkey())]["label"] == f"wallet-{i}"
    assert "secret" not in json.dumps(pub)


def test_import_labels_the_old_wallets_from_the_secret(vault, monkeypatch):
    from solders.keypair import Keypair

    old = [Keypair() for _ in range(2)]
    monkeypatch.setenv("TB_CANARY_WALLETS", json.dumps([list(bytes(k)) for k in old]))
    keyvault.cmd_import()
    pub = json.load(open(vault / "addressbook.json"))
    for k in old:
        assert pub["addresses"][str(k.pubkey())]["label"].startswith("wallet-old-")


def test_sync_rebuilds_the_addressbook(vault):
    from solders.keypair import Keypair

    keyvault.record("canary", Keypair())
    os.remove(vault / "addressbook.json")
    keyvault.cmd_sync()
    assert json.load(open(vault / "addressbook.json"))["count"] == 1


def test_cli_keyvault_list_shows_addresses_not_secrets(vault, capsys):
    from solders.keypair import Keypair

    from tradingbot import cli

    kp = Keypair()
    keyvault.record("wallet-0", kp)
    cli.main(["keyvault", "list"])
    out = capsys.readouterr().out
    assert str(kp.pubkey()) in out and "wallet-0" in out and "total: 1" in out
    assert json.dumps(list(bytes(kp))) not in out  # the secret int-array never prints
