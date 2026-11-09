"""The real-money launch canary: the live executor's fills read from the transaction, and the book's limits."""

import base64
import time
from types import SimpleNamespace

import pytest

from tradingbot import launch_canary
from tradingbot import model_book as mb
from tradingbot.config import LIVE_CONFIRM_PHRASE, Config
from tradingbot.data.jupiter import USDC_MINT, Quote
from tradingbot.engine import CANARY_HALT, Engine
from tradingbot.execution import jupiter as jx
from tradingbot.execution.paper import PaperExecutor
from tradingbot.journal import Journal
from tests.test_analyst_engine import FakeDex

solders = pytest.importorskip("solders")
from solders.hash import Hash  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from solders.message import MessageV0  # noqa: E402
from solders.transaction import VersionedTransaction  # noqa: E402

TOKEN = "Tok1111111111111111111111111111111111111111"


class FakeJup:
    def __init__(self, impact=0.5):
        self.impact = impact
        self.http = self
        self.posts = []

    def decimals(self, mint):
        return 6

    def sol_price(self):
        return 200.0

    def last_failure(self):
        return "no route"

    def quote(self, in_mint, out_mint, in_raw, slippage_bps=None):
        self.last_slippage = slippage_bps
        return Quote(in_mint, out_mint, in_raw, in_raw * 2, in_raw, self.impact, ["x"], {"q": 1})

    def post(self, path, json=None):  # /swap/v1/swap
        self.posts.append(json)
        from solders.pubkey import Pubkey
        from solders.null_signer import NullSigner
        payer = Pubkey.from_string(json["userPublicKey"])
        tx = VersionedTransaction(MessageV0.try_compile(payer, [], [], Hash.default()), [NullSigner(payer)])
        return SimpleNamespace(status_code=200, json=lambda: {"swapTransaction": base64.b64encode(bytes(tx)).decode()},
                               text="")


class FakeRpc:
    """Answers the executor's JSON-RPC calls from a script of balances and transactions."""

    def __init__(self, owner, usdc=50_000_000, token=0, sol=1_000_000_000):
        self.owner, self.bal, self.sol = owner, {USDC_MINT: usdc, TOKEN: token}, sol
        self.txs, self.sent, self.status, self.send_error = {}, [], {}, None
        self.next_meta = None
        self.raw_sig, self.held, self.resends, self.land_on_resend, self.height = {}, {}, 0, False, None

    def post(self, url, json=None):
        m, p = json["method"], json["params"]
        result, error = None, None
        if m == "getTokenAccountsByOwner":
            amt = self.bal.get(p[1]["mint"], 0)
            result = {"value": [] if amt is None else [{"pubkey": "Acc1111111111111111111111111111111111111111",
                                                        "account": {"owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                                                                    "data": {"parsed": {"info": {"tokenAmount": {"amount": str(amt)}}}}}}]}
        elif m == "getBalance":
            result = {"value": self.sol}
        elif m == "sendTransaction":
            if p[0] in self.raw_sig:  # the same signed bytes again: same signature
                self.resends += 1
                result = self.raw_sig[p[0]]
                if self.land_on_resend:
                    self.txs[result] = self.held.pop(result, self.txs.get(result))
            elif self.send_error:
                error = {"message": self.send_error}
            else:
                sig = f"sig{len(self.sent)}"
                self.sent.append(sig)
                self.txs[sig] = self.next_meta.pop(0) if isinstance(self.next_meta, list) else self.next_meta
                self.raw_sig[p[0]] = sig
                if self.land_on_resend:  # the first send is lost; it lands only once sent again
                    self.held[sig], self.txs[sig] = self.txs[sig], None
                result = sig
        elif m == "getBlockHeight":
            result = self.height
        elif m == "getSignatureStatuses":
            sig = p[0][0]
            meta = self.txs.get(sig)
            result = {"value": [None if meta is None else {"confirmationStatus": "confirmed", "err": meta.get("err")}]}
        elif m == "getTransaction":
            meta = self.txs.get(p[0])
            result = None if meta is None else {"meta": meta}
        elif m == "getLatestBlockhash":
            result = {"value": {"blockhash": str(Hash.default())}}
        body = {"error": error} if error else {"result": result}
        return SimpleNamespace(json=lambda: body)


def meta(owner, usdc, token, lamports_spent=0.001e9, err=None):
    def rows(u, t):
        return [{"mint": USDC_MINT, "owner": owner, "uiTokenAmount": {"amount": str(u)}},
                {"mint": TOKEN, "owner": owner, "uiTokenAmount": {"amount": str(t)}},
                {"mint": TOKEN, "owner": "someone-else", "uiTokenAmount": {"amount": "999"}}]
    return {"err": err, "preBalances": [int(1e9)], "postBalances": [int(1e9 - lamports_spent)],
            "preTokenBalances": rows(usdc[0], token[0]), "postTokenBalances": rows(usdc[1], token[1])}


def executor(**kw):
    kp = Keypair()
    cfg = Config(max_price_impact_pct=5.0)
    rpc = FakeRpc(str(kp.pubkey()), **kw)
    ex = jx.JupiterExecutor(cfg, FakeJup(), http=rpc, keypair=kp, sell_slippage_bps=1000, sleep=lambda s: None)
    return ex, rpc


def test_buy_reads_the_fill_from_the_transaction():
    ex, rpc = executor()
    rpc.next_meta = meta(ex.owner, (50_000_000, 40_100_000), (0, 19_000_000), lamports_spent=0.003e9)
    fill = ex.buy(TOKEN, 10.0, 0.5)
    assert fill.ok and fill.qty == pytest.approx(19.0)  # other wallets' balance rows are ignored
    assert fill.fee_usd == pytest.approx(0.6)  # fee + rent: 0.003 SOL at $200
    assert fill.price == pytest.approx(9.9 / 19.0) and fill.usd == pytest.approx(10.5)
    assert rpc.sent == ["sig0"]


def test_buy_declines_before_sending_when_the_wallet_is_short():
    ex, rpc = executor(usdc=5_000_000)
    assert ex.buy(TOKEN, 10.0, 0.5) is None and "USDC" in ex.last_note and not rpc.sent
    ex, rpc = executor(sol=1_000_000)
    assert ex.buy(TOKEN, 10.0, 0.5) is None and "SOL" in ex.last_note and not rpc.sent


def test_preflight_refusal_sends_nothing_and_a_revert_costs_its_fee():
    ex, rpc = executor()
    rpc.send_error = "slippage tolerance exceeded"
    assert ex.buy(TOKEN, 10.0, 0.5) is None and ex.last_note.startswith(jx.NOT_SENT)
    rpc.send_error = None
    rpc.next_meta = meta(ex.owner, (50_000_000, 50_000_000), (0, 0), err={"InstructionError": [3, 1]})
    fill = ex.buy(TOKEN, 10.0, 0.5)
    assert not fill.ok and "reverted" in fill.note and fill.usd == pytest.approx(0.2)


def test_dropped_transaction_costs_nothing(monkeypatch):
    monkeypatch.setattr(jx, "CONFIRM_TIMEOUT_S", 0)
    ex, rpc = executor()
    rpc.next_meta = None  # never lands
    fill = ex.buy(TOKEN, 10.0, 0.5)
    assert not fill.ok and "dropped" in fill.note and fill.usd == 0


def test_a_lost_send_is_sent_again_until_it_lands(monkeypatch):
    monkeypatch.setattr(jx, "RESEND_S", 0)
    ex, rpc = executor()
    rpc.land_on_resend = True
    rpc.next_meta = meta(ex.owner, (50_000_000, 40_100_000), (0, 19_000_000))
    fill = ex.buy(TOKEN, 10.0, 0.5)
    assert fill.ok and rpc.resends >= 1 and rpc.sent == ["sig0"]  # one signature, so it can only land once


def test_expired_blockhash_stops_the_resends(monkeypatch):
    monkeypatch.setattr(jx, "RESEND_S", 0)
    ex, rpc = executor()
    rpc.next_meta, rpc.height = None, 10**12  # never lands; the chain is past any last valid height
    orig = ex.jup.post
    ex.jup.post = lambda path, json=None: SimpleNamespace(
        status_code=200, text="", json=lambda: {**orig(path, json).json(), "lastValidBlockHeight": 5})
    fill = ex.buy(TOKEN, 10.0, 0.5)
    assert not fill.ok and fill.note == "not confirmed (dropped) sig0" and rpc.resends == 0


def test_sell_uses_wide_slippage_and_closes_the_empty_account():
    ex, rpc = executor(token=19_000_000)
    sold = meta(ex.owner, (0, 12_000_000), (19_000_000, 0), lamports_spent=0.0005e9)
    closed = meta(ex.owner, (12_000_000, 12_000_000), (0, 0), lamports_spent=-0.002e9)  # rent back
    rpc.next_meta = [sold, closed]
    orig = rpc.post

    def post(url, json=None):  # after the sale the token account is empty
        if json["method"] == "getTokenAccountsByOwner" and rpc.sent:
            rpc.bal[TOKEN] = 0
        return orig(url, json)
    rpc.post = post
    fill = ex.sell(TOKEN, 19.0, 0.6)
    assert ex.jup.last_slippage == 1000
    assert fill.ok and fill.qty == pytest.approx(19.0)
    assert len(rpc.sent) == 2  # the sale, then the account close
    assert fill.usd == pytest.approx(12.0 - 0.1 + 0.4)  # proceeds - sale fee + rent back


def test_sell_takes_the_rounding_dust_too():
    ex, rpc = executor(token=12_345_678_901)
    rpc.next_meta = [meta(ex.owner, (0, 12_000_000), (12_345_678_901, 0)), meta(ex.owner, (0, 0), (0, 0))]
    sent = []
    orig = ex.jup.quote
    ex.jup.quote = lambda a, b, raw, slip=None: (sent.append(raw), orig(a, b, raw, slip))[1]
    ex.sell(TOKEN, 12_345_678_900 / 1e6, 0.6)  # one raw unit short of the balance (rounding): sell it all
    assert sent == [12_345_678_901]  # a single unit left would have kept the account (and its rent) open


def test_a_dropped_account_close_is_built_again(monkeypatch):
    monkeypatch.setattr(jx, "CONFIRM_TIMEOUT_S", 0)
    ex, rpc = executor()
    rpc.next_meta = [None, meta(ex.owner, (0, 0), (0, 0), lamports_spent=-0.002e9)]  # dropped, then lands
    orig = rpc.post

    def post(url, json=None):  # a fresh blockhash each time, so the rebuilt close is a new transaction
        if json["method"] == "getLatestBlockhash":
            return SimpleNamespace(json=lambda: {"result": {"value": {"blockhash": str(Hash.new_unique()),
                                                                       "lastValidBlockHeight": 99}}})
        return orig(url, json)
    rpc.post = post
    rpc.bal[TOKEN] = 0
    assert ex.close_empty(TOKEN) == pytest.approx(0.4)  # 0.002 SOL rent back at $200
    assert len(rpc.sent) == 2


def test_sweep_closes_empty_accounts_but_not_usdc_or_open_positions():
    ex, rpc = executor()
    def acct(key, mint, amount, program=jx.TOKEN_PROGRAMS[0]):
        return {"pubkey": key, "account": {"owner": program, "lamports": 1_500_000, "data": {"parsed": {"info": {
            "mint": mint, "tokenAmount": {"amount": str(amount)}}}}}}
    keys = [str(Keypair().pubkey()) for _ in range(12)]
    wallet = {jx.TOKEN_PROGRAMS[0]: [acct(keys[0], USDC_MINT, 0), acct(keys[1], "WsolMint", 0)],
              jx.TOKEN_PROGRAMS[1]: [acct(keys[2 + i], f"Dead{i}", 0, jx.TOKEN_PROGRAMS[1]) for i in range(8)]
              + [acct(keys[10], "OpenMint", 0, jx.TOKEN_PROGRAMS[1]), acct(keys[11], "Left", 5, jx.TOKEN_PROGRAMS[1])]}
    rpc.next_meta = [meta(ex.owner, (0, 0), (0, 0), lamports_spent=-0.012e9),
                     meta(ex.owner, (0, 0), (0, 0), lamports_spent=-0.0015e9)]
    orig = rpc.post

    def post(url, json=None):
        if json["method"] == "getTokenAccountsByOwner":
            return SimpleNamespace(json=lambda: {"result": {"value": wallet[json["params"][1]["programId"]]}})
        return orig(url, json)
    rpc.post = post
    listed = ex.sweep({"OpenMint"}, apply=False)
    assert listed["empty_accounts"] == 9 and listed["rent_sol"] == pytest.approx(0.0135) and not rpc.sent
    assert listed["holding_tokens"] == [{"account": keys[11], "mint": "Left", "amount": 5}]
    res = ex.sweep({"OpenMint"})
    assert res["closed"] == 9 and len(rpc.sent) == 2  # 8 per transaction
    assert res["usd_back"] == pytest.approx(2.7)


def test_sell_not_sent_is_retried_next_pass(cfg, good_pair):
    j = Journal(":memory:")
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j)
    from tradingbot.models import Position
    pos = Position(None, "TOK", "T", 1.0, 10.0, 10.0, 10.0, time.time(), 0.0, 1.0, strategy="claude")
    j.save_position(pos)
    book = eng.books["claude"]
    book.executor = SimpleNamespace(sell=lambda *a: jx.failed(0.0, jx.NOT_SENT + " preflight"))
    assert not eng._sell(book, pos, 10.0, 1.0, "time_stop")
    assert pos.id not in eng._blocked  # not treated as a pool with no route


def test_config_needs_the_confirmation_phrase():
    with pytest.raises(ValueError):
        Config(launch_canary=True).validate()
    Config(launch_canary=True, live_confirm=LIVE_CONFIRM_PHRASE).validate()
    with pytest.raises(ValueError):
        Config(launch_canary=True, live_confirm=LIVE_CONFIRM_PHRASE, mode="live").validate()


class Recorder(PaperExecutor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sizes = []

    def buy(self, token, usd, ref_price):
        self.sizes.append(usd)
        return super().buy(token, usd, ref_price)


def test_canary_book_takes_top2_picks_at_its_own_size_and_limits(cfg, good_pair):
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = True
    try:
        j = Journal(":memory:")
        args = (cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j)
        assert mb.CANARY4 not in Engine(*args, model_executor=PaperExecutor(cfg)).books  # no live executor: no book
        live = Recorder(cfg)
        eng = Engine(*args, model_executor=PaperExecutor(cfg), launch_canary_executor=live)
        book = eng.books[mb.CANARY4]
        assert book.executor is live and book.cash == cfg.launch_canary_budget_usd
        created = time.time() - 6

        def sig(mint, top2=True, liq=12_000.0):  # the canary follows launchrw50_sz's flag, thin pools only
            return {"pool": "P" + mint, "mint": mint, "dex": "pumpswap", "created_ts": created, "name": "", "price": 1e-5,
                    "liq": liq, "score": 1.0, "ts": time.time(), "decided_ts": time.time(), "kind": "launch",
                    "rw50": top2, "rw50_24": not top2, "top2": not top2, "top5": True}
        eng.model_book.signals.put(sig("A"))
        eng.model_book.signals.put(sig("B", top2=False))
        eng.model_book.signals.put(sig("D", liq=20_000.0))
        eng.buy_model_signals()
        assert [p.token_address for p in j.open_positions(mb.CANARY4)] == ["A"]  # the twin's thin-pool picks only
        assert eng._model_skip_reason(book, sig("E", liq=20_000.0)) == "canary trades thin pools only (< $15,000)"
        assert live.sizes == [mb.band_usd(mb.CANARY4, 10_000.0)] == [25.0]  # band size, not the fixed fallback
        assert eng._deadlines[j.open_positions(mb.CANARY4)[0].id] == created + 360
        j.set_state(CANARY_HALT, "1")  # kill switch
        assert eng._model_skip_reason(book, sig("C")) == "canary halted (kill switch)"
        j.execute("DELETE FROM state WHERE key=?", (CANARY_HALT,))
        real = j.realized_since   # the dd24 paper twin down $500 over 24 h pauses the canary; its own losses don't
        j.realized_since = lambda since, name: -500.0 if name == mb.LAUNCH_SIZED_DD else 0.0
        assert eng._model_skip_reason(book, sig("C")) == \
            "twin launchrw50_sz_dd24 paused: past 24h drawdown limit (down $500)"
        cfg.launch_canary_follow_dd = False
        assert eng._model_skip_reason(book, sig("C")) is None
        cfg.launch_canary_follow_dd = True
        j.realized_since = lambda since, name: -499.0 if name == mb.LAUNCH_SIZED_DD else -40.0
        assert eng._model_skip_reason(book, sig("C")) is None
        j.realized_since = real
        cfg.launch_canary_max_open = 1
        assert eng._model_skip_reason(book, sig("C")) == "max open positions"
        cfg.launch_canary_max_open = 5
        cfg.launch_canary_every_n = 3  # sampling: 1 in 3 signals goes through
        reasons = [eng._model_skip_reason(book, sig(f"S{i}")) for i in range(6)]
        assert reasons.count(None) == 2 and reasons.count("canary samples 1 in 3 signals") == 4
        cfg.launch_canary_every_n = 1
        cfg.launch_canary_daily_loss_usd = 0.0
        assert eng._model_skip_reason(book, sig("C")) == "daily loss limit"
        st = launch_canary.status(j, cfg)
        assert st["open_positions"] == 1 and st["enabled"]
    finally:
        cfg.launch_book = cfg.launch_live = cfg.launch_canary = False
        cfg.launch_canary_daily_loss_usd, cfg.launch_canary_max_open = 50.0, 5


def test_calibration_pairs_real_and_simulated_trades():
    from tradingbot.models import Position
    j = Journal(":memory:")
    now = time.time()
    for book, buy_px, sell_px, fee in ((mb.CANARY4, 1.02, 1.10, 0.4), (mb.CANARY_TWIN, 1.00, 1.15, 0.3)):
        cost = 10.0 if book == mb.CANARY4 else 100.0
        qty = (cost - fee / 2) / buy_px
        proceeds = qty * sell_px - fee / 2
        pos = Position(None, "TOK", "T", buy_px, 0.0, qty, cost, now, 0.0, buy_px, realized_usd=proceeds,
                       closed_at=now + 360, exit_reason="pool_age_exit", strategy=book)
        j.save_position(pos)
        j.log_fill(pos.id, "buy", qty, buy_px, cost, fee / 2, "entry", "x")
        j.log_fill(pos.id, "sell", qty, sell_px, proceeds, fee / 2, "pool_age_exit", "x")
    out = launch_canary.calibration(j)
    assert out["paired"] == 1
    p = out["pairs"][0]
    assert p["entry_vs_sim_pct"] == pytest.approx(2.0) and p["gap"] < 0
    assert p["swap_gap"] == pytest.approx((1.10 / 1.02 - 1.15) * 100, abs=0.01)


def test_keypair_loads_from_json_array_or_base58():
    kp = Keypair()
    assert jx.load_keypair(str(list(bytes(kp)))).pubkey() == kp.pubkey()
    assert jx.load_keypair(f"  {kp}\n").pubkey() == kp.pubkey()  # base58, as Phantom exports it
    assert jx.load_keypair(f'"{kp}"').pubkey() == kp.pubkey()


def test_keypair_is_found_without_the_entrypoint(tmp_path, monkeypatch):
    """`fly machine exec` skips the entrypoint: no TB_KEYPAIR_PATH, but the secret file or the secret itself."""
    kp = Keypair()
    cfg = Config(db_path=":memory:", keypair_path=str(tmp_path / "missing.json"))
    monkeypatch.setattr(jx, "SECRET_KEYPAIR_PATH", str(tmp_path / "secret.json"))
    monkeypatch.delenv("TB_KEYPAIR_JSON", raising=False)
    with pytest.raises(FileNotFoundError):
        jx.read_keypair(cfg)
    monkeypatch.setenv("TB_KEYPAIR_JSON", str(kp))
    assert jx.read_keypair(cfg).pubkey() == kp.pubkey()
    other = Keypair()
    (tmp_path / "secret.json").write_text(str(list(bytes(other))))
    assert jx.read_keypair(cfg).pubkey() == other.pubkey()  # the entrypoint's copy comes first


def test_canary_goes_below_the_paper_books_five_dollar_floor(cfg, good_pair):
    # 11-06: `_buy_model` refused anything under $5, so the $0.50 / $2 bands never traded; the canary may go to a cent
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = cfg.launch_sized_book = True
    cfg.launch_canary_max_liq_usd = 0.0
    j = Journal(":memory:")
    args = (cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j)
    live = Recorder(cfg)
    eng = Engine(*args, model_executor=PaperExecutor(cfg), launch_canary_executor=live)
    created = time.time() - 6

    def sig(mint, liq):
        return {"pool": "P" + mint, "mint": mint, "dex": "pumpswap", "created_ts": created, "name": "", "price": 1e-5,
                "liq": liq, "score": 1.0, "ts": time.time(), "decided_ts": time.time(), "kind": "launch",
                "rw50": True, "rw50_24": False, "top2": False, "top5": True}
    eng.model_book.signals.put(sig("MID", 20_000.0))  # $5 band (11-10)
    eng.model_book.signals.put(sig("DEEP", 60_000.0))  # $10 band
    eng.buy_model_signals()
    assert sorted(live.sizes) == [5.0, 10.0]
    cfg.launch_canary_flat_usd = 0.01  # one minimum size for every pick
    eng.model_book.signals.put(sig("THIN", 10_000.0))
    eng.buy_model_signals()
    assert live.sizes[-1] == 0.01
    # the paper twin keeps its own sizes and the $5 floor
    assert eng._buy_model(eng.books[mb.LAUNCH_SIZED], sig("X", 100.0)) == "size too small (cash or liquidity)"


def test_engine_sweeps_the_canary_wallet_hourly_keeping_open_positions(cfg, good_pair):
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = True
    j = Journal(":memory:")
    live = Recorder(cfg)
    swept = []
    live.sweep = lambda keep: swept.append(set(keep)) or {"closed": 2, "usd_back": 0.6}
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 model_executor=PaperExecutor(cfg), launch_canary_executor=live)
    from tradingbot.models import Position
    j.save_position(Position(None, "HELD", "H", 1.0, 1.0, 1.0, 0.3, time.time(), 0.0, 1.0, strategy=mb.CANARY4))
    eng.sweep_canary_wallet_if_due(wait=True)
    eng.sweep_canary_wallet_if_due(wait=True)  # not due again for an hour
    assert swept == [{"HELD"}]
    assert j.query("SELECT kind FROM events WHERE kind='wallet_sweep'")


def test_canary_can_trade_a_fraction_of_the_twins_band_sizes(cfg, good_pair):
    # 11-07: user's live setup = 1/25 of launchrw50_sz's $250 / $25 / $100, every pool
    cfg.launch_book = cfg.launch_live = cfg.launch_canary = cfg.launch_sized_book = True
    cfg.launch_canary_max_liq_usd, cfg.launch_canary_band_scale, cfg.launch_canary_budget_usd = 0.0, 0.04, 600.0
    cfg.launch_canary_max_open = 20
    j = Journal(":memory:")
    live = Recorder(cfg)
    eng = Engine(cfg, FakeDex(good_pair), SimpleNamespace(report=lambda m: None), None, PaperExecutor(cfg), j,
                 model_executor=PaperExecutor(cfg), launch_canary_executor=live)
    created = time.time() - 6

    def sig(mint, liq):
        return {"pool": "P" + mint, "mint": mint, "dex": "pumpswap", "created_ts": created, "name": "", "price": 1e-5,
                "liq": liq, "score": 1.0, "ts": time.time(), "decided_ts": time.time(), "kind": "launch",
                "rw50": True, "rw50_24": False, "top2": False, "top5": True}
    for mint, liq in (("THIN", 14_000.0), ("MID", 20_000.0), ("DEEP", 60_000.0)):
        eng.model_book.signals.put(sig(mint, liq))
    eng.buy_model_signals()
    assert sorted(round(x, 2) for x in live.sizes) == [1.0, 4.0, 10.0]
