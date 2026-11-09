"""dd-limits: daily and trailing drawdown limits replayed on a book's real trades and on the year replay (11-09)."""
from tradingbot import dd_limits as dl
from tradingbot.journal import Journal

D = 86400.0
T0 = 1_780_000_000.0 // D * D  # midnight UTC


def _t(opened, ret, cost=1000.0, hold=900.0):
    return {"opened": opened, "ts": opened + hold, "cost": cost, "ret": ret}


def _day1():
    """Three -50% trades on $1,000 (overlapping holds), then ten +10%; day 2: ten +10%."""
    tr = [_t(T0 + i * 600, -0.5) for i in range(3)]
    tr += [_t(T0 + 3600 + i * 600, 0.10) for i in range(10)]
    tr += [_t(T0 + D + i * 600, 0.10) for i in range(10)]
    return tr


def test_day_loss_stops_the_rest_of_the_day_counting_only_trades_already_sold():
    tr = _day1()
    m = dl.apply_rule(tr, {"kind": "day", "k": 10}, dl.CAPITAL)          # $1,000
    assert m[:3] == [1.0, 1.0, 1.0]       # the second and third were bought before -$1,000 was realized
    assert m[3:13] == [0.0] * 10          # day 1 stopped at -$1,500
    assert m[13:] == [1.0] * 10           # a new day trades again
    half = dl.apply_rule(tr, {"kind": "day", "k": 10, "half": True}, dl.CAPITAL)
    assert half[3:13] == [0.5] * 10


def test_day_trail_reacts_to_a_give_back_from_the_days_high():
    tr = [_t(T0 + i * 600, 0.5, hold=300) for i in range(4)]              # +$2,000 by 00:35
    tr += [_t(T0 + 3000 + i * 600, -0.5, hold=300) for i in range(3)]     # give back $1,500
    tr += [_t(T0 + 6000 + i * 600, 0.1, hold=300) for i in range(3)]
    day = dl.apply_rule(tr, {"kind": "day", "k": 10}, dl.CAPITAL)
    trail = dl.apply_rule(tr, {"kind": "trail", "k": 10}, dl.CAPITAL)
    assert day == [1.0] * 10                       # still up on the day: a loss limit never fires
    assert trail[:6] == [1.0] * 6 and trail[6:] == [0.0] * 4   # $1,000 off the high after two losers


def test_pause_restarts_after_its_hours_and_peak_trail_restarts_the_next_day():
    tr = [_t(T0 + i * 600, -0.5, hold=300) for i in range(3)]             # -$1,500 by 00:25
    tr += [_t(T0 + 2 * 3600 + i * 600, 0.1, hold=300) for i in range(3)]  # 02:00
    tr += [_t(T0 + 5 * 3600, 0.1, hold=300)]                              # 05:00
    tr += [_t(T0 + D + 3600, 0.1, hold=300)]                              # next day
    p = dl.apply_rule(tr, {"kind": "pause", "h": 3, "k": 10}, dl.CAPITAL)
    assert p[2] == 0.0 and p[3:6] == [0.0] * 3 and p[6] == 1.0            # paused 00:20..03:20
    pk = dl.apply_rule(tr, {"kind": "peak", "k": 10}, dl.CAPITAL)
    assert pk[6] == 0.0 and pk[7] == 1.0
    roll = dl.apply_rule(tr, {"kind": "roll", "h": 3, "k": 10}, dl.CAPITAL)
    assert roll[3:6] == [0.0] * 3 and roll[6] == 1.0                     # losses left the 3 h window


def test_study_scores_every_limit_against_no_limit():
    j = Journal(":memory:")
    for t in _day1():
        j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                  "VALUES(?,?,?,?,?,?)", ("X", t["cost"], t["opened"], t["ts"], t["cost"] * (1 + t["ret"]), dl.BOOK))
    out = dl.study(j, now=T0 + 3 * D)
    rs = {r["limit"]: r for r in out["life"]["rules"]}
    assert rs["no limit"]["usd"] == 500 and rs["day loss 10%"]["usd"] == -500
    assert rs["day loss 20%"]["usd"] == 500        # never reached
    assert len(rs) == len(dl.rules())
    text = dl.render(out)
    assert "BEST 10 BY $ PER $ OF DRAWDOWN" in text and "-- pause trail" in text


def test_year_trades_are_rebuilt_full_size_and_limits_scaled(monkeypatch):
    from tradingbot import risk_history as rh
    from tradingbot import safeguards as sg
    rows = [{"create_ts": T0 - 40 * D + i * 3000, "ret": 0.05 if i % 3 else -0.6, "liq_usd": 10_000.0, "rw": 0.1,
             "rw_pct": 10.0, "rwb_pct": 10.0, "w": 1.0} for i in range(1000)]
    monkeypatch.setattr(sg, "load_replay", lambda store: (rows, {}))
    monkeypatch.setattr(rh, "gaps", lambda *a: {dl.BOOK: {"gap": 0.02}})

    class Cfg:
        model_book_liq_share = 0.02
    j = Journal(":memory:")
    for t in _day1():
        j.execute("INSERT INTO positions(token, cost_usd, opened_at, closed_at, realized_usd, strategy) "
                  "VALUES(?,?,?,?,?,?)", ("X", t["cost"], t["opened"], t["ts"], t["cost"] * (1 + t["ret"]), dl.BOOK))
    out = dl.study(j, now=T0 + 3 * D, year=True, hist_store=object(), watch_store=None, cfg=Cfg())
    y = out["year"]
    assert y["trades"] == 1000 and y["scale"] > 0 and "months" in y["rules"][0]
    assert "YEAR REPLAY" in dl.render(out)


def test_cli_runs_without_data(cfg, tmp_path, capsys, monkeypatch):
    from tradingbot import cli
    monkeypatch.setenv("TB_DB_PATH", str(tmp_path / "j.db"))
    cli.main(["dd-limits", "--year"])
    out = capsys.readouterr().out
    assert "DRAWDOWN LIMITS" in out and "no hist.db" in out
