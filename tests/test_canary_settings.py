import json
import os

import pytest

from tradingbot import canary_settings as cs
from tradingbot.config import Config


def _cfg(tmp_path):
    return Config(db_path=str(tmp_path / "j.db"))


def test_validate_coerces_types_and_checks_bounds():
    out = cs.validate({"launch_canary_feint": "on", "launch_canary_max_open": "7", "launch_canary_route": "direct",
                       "launch_canary_flat_usd": "5"})
    assert out == {"launch_canary_feint": True, "launch_canary_max_open": 7, "launch_canary_route": "direct",
                   "launch_canary_flat_usd": 5.0}
    with pytest.raises(ValueError):
        cs.validate({"launch_canary_max_open": 999})      # above the bound
    with pytest.raises(ValueError):
        cs.validate({"launch_canary_route": "sideways"})  # not a choice
    with pytest.raises(ValueError):
        cs.validate({"not_a_setting": 1})                 # unknown key


def test_apply_sets_live_fields_now_persists_and_flags_restart(tmp_path):
    cfg = _cfg(tmp_path)
    res = cs.apply(cfg, None, {"launch_canary_max_open": 8, "launch_canary_route": "direct",
                               "launch_canary_budget_usd": 200})
    assert cfg.launch_canary_max_open == 8 and cfg.launch_canary_route == "direct" and cfg.launch_canary_budget_usd == 200
    assert "launch_canary_max_open" not in res["restart_needed"]          # live field
    assert set(res["restart_needed"]) == {"launch_canary_route", "launch_canary_budget_usd"}
    assert set(res["changed"]) == {"launch_canary_max_open", "launch_canary_route", "launch_canary_budget_usd"}
    overlay = cs.load_overlay(cfg)
    assert overlay["launch_canary_max_open"] == 8 and overlay["launch_canary_route"] == "direct"


def test_coherent_rejects_feint_without_the_direct_route(tmp_path):
    cfg = _cfg(tmp_path)  # default route is jupiter
    with pytest.raises(ValueError):
        cs.apply(cfg, None, {"launch_canary_feint": True})
    with pytest.raises(ValueError):
        cs.apply(cfg, None, {"launch_canary_feint_all": True, "launch_canary_route": "direct"})  # feint_all needs feint
    assert cfg.launch_canary_feint is False  # a rejected apply changes nothing


def test_preset_applies_a_coherent_set(tmp_path):
    cfg = _cfg(tmp_path)
    cs.apply(cfg, None, cs.PRESETS["clips"]["values"])
    assert cfg.launch_canary_clips == 5 and cfg.launch_canary_route == "direct" and cfg.launch_canary_flat_usd == 5.0
    cs.apply(cfg, None, cs.PRESETS["feint_measure"]["values"])
    assert cfg.launch_canary_feint and cfg.launch_canary_feint_only and cfg.launch_canary_feint_all
    assert cfg.launch_canary_max_liq_usd == 0.0


def test_apply_overlay_is_defensive_at_boot(tmp_path):
    cfg = _cfg(tmp_path)
    with open(cs.settings_path(cfg), "w") as fh:
        json.dump({"launch_canary_max_open": 9, "launch_canary_every_n": 99999999, "bogus": 1}, fh)
    applied = cs.apply_overlay(cfg)
    assert cfg.launch_canary_max_open == 9            # good value applied
    assert "launch_canary_every_n" not in applied     # out-of-bounds skipped, no crash
    assert "bogus" not in applied


def test_reset_clears_the_overlay_and_restores_env_values(tmp_path):
    cfg = _cfg(tmp_path)
    base_open = cfg.launch_canary_max_open
    cs.apply(cfg, None, {"launch_canary_max_open": 11})
    assert os.path.exists(cs.settings_path(cfg))
    cs.reset(cfg)
    assert not os.path.exists(cs.settings_path(cfg)) and cfg.launch_canary_max_open == base_open


def test_menu_has_fields_groups_current_and_presets(tmp_path):
    m = cs.menu(_cfg(tmp_path))
    assert set(m) == {"fields", "groups", "current", "presets"}
    assert all(f["key"] in m["current"] for f in m["fields"])
    assert "clips" in m["presets"] and set(m["presets"]["clips"]) == {"label", "about"}
