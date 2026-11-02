"""Pumpswap account parsing on real mainnet bytes (captured 2026-11-02)."""
import base64
import pytest
from tradingbot.data import pumpswap as ps

GRAD = base64.b64decode("8ZptBBGxbbz/AACTl5brgYlUCFwW746YfwBwxSNXoSsEHMgwvygDrwsRKubljRFJK6lz+/n8+4D/ucTD0BrG40BaAEwKnFceFWjPBpuIV/6rgYT7aH9jRhjANdrEOdwa6ztVmKDwAAAAAAE60D4oNTc3qv5Kob/Vl6Cdo/hGJmuWkUo0INxSsPlg09dfv2peAu6hltUtkS2EtgK5mCUHWxp8JWKZU2B8TaKKuSUFcwt7iwM1Hw9/CUdpOkddJa7fgKunSwva8YjJ2VfsQmtZ0AMAAJwh3vKQpFqGDQ3zSCQlDAcf34XJgDbIYYT0iGBHZtTYAABh8QwYBAAAAAAAAAAAAAAAAAAAAAAAAAAAADSGCgAAAAAAM8oGAAAAAAAAAAAAAAAAAAAAAAAAAA==")
HAND = base64.b64decode("8ZptBBGxbbz7AAD7TurfQ1jgslQAoWeXP86RceRM/D9ejLIsMZ0W+YSawQabiFf+q4GE+2h/Y0YYwDXaxDncGus7VZig8AAAAAABUKPfmmEXdUHLn7VOIrXd1RqpzDcRBg4tqS8mTITN8tUlG1XZ70SL0KO72wPVtwpV2tf+ELXgmgiVLt1G/dEciwtpz1D0afKAwGQ6W5Z4e8MWvoeLpO+qZYkCcQnDQbiuOTVR9Sq84br3ZOEibXAaQqaCcpI8wt9m77xAleXcPPtkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbGFv+////////////////AAAAAAAAAAAAAOXnpAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==")
LP = base64.b64decode("AQAAAEzy68OdC3ToOQjmBTNqDD1mWZgmVmNffVVkU8mzuNlYAAAAAAAAAAAJAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==")


def test_graduation_pool_carries_the_coin_creator():
    p = ps.parse_pool(GRAD)
    assert p["handmade"] is False and p["coin_creator"] == "BWUZGrS4orCq7sQEF5Kn9wWVEU8CgG9xxs6tjj9WLKqh"
    assert p["token_mint"] == p["base_mint"] == "GYKnd6jGUSQoBUnj6LdGCBtCWYDgr2Zd6QwNPcy5pump"
    assert p["quote_mint"] == ps.WSOL and p["sol_vault"] == p["quote_vault"]
    assert p["lp_mint"] == "4xapQw4x5o8ncfG7JXBWFg9GZ4grCiHpJZDx5kKXm2kr"


def test_handmade_pool_with_sol_as_base():
    p = ps.parse_pool(HAND)
    assert p["handmade"] is True and p["coin_creator"] == ps.NO_KEY
    assert p["base_mint"] == ps.WSOL and p["token_mint"] == p["quote_mint"] == "6RnYgrbQL7HBDyB16VW9zvi1hTrTR9vowxsf29eA6Ks2"
    assert p["sol_vault"] == p["base_vault"] and p["token_vault"] == p["quote_vault"]
    assert p["creator"] == "Hv18Nj5uNvNJFx8hGk5oVhgnbD8nmt4DpGfsTpAWPofE"


def test_lp_mint_authority_is_the_pool_and_pulled_supply_is_zero():
    m = ps.parse_lp_mint(LP)
    assert m == {"authority": "6BNptAEKGETQFGQgDWiDhFqAgAmHusyqXqJnLLw74Exw", "supply": 0, "decimals": 9}
    assert ps.parse_lp_mint(LP[:-1]) is None


def test_non_sol_pair_and_short_account_are_rejected():
    assert ps.parse_pool(GRAD[:200]) is None
    no_sol = bytearray(GRAD)
    no_sol[75:107] = bytes(32)  # quote mint no longer wSOL
    assert ps.parse_pool(bytes(no_sol)) is None


def test_token_amount_and_spot_price():
    acct = bytes(64) + (5_000_000).to_bytes(8, "little") + bytes(93)
    assert ps.token_amount(acct) == 5_000_000 and ps.token_amount(bytes(10)) is None
    assert ps.spot_price(2_000_000_000, 1_000_000 * 10 ** 6, 6) == pytest.approx(2e-6)
    assert ps.spot_price(0, 1, 6) is None
