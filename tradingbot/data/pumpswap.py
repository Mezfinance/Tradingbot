"""Pumpswap accounts, read straight from their bytes (no Anchor IDL needed).

Pool account (301 bytes; some older pools 287): 8-byte discriminator, bump u8, index u16, then
creator @11, base_mint @43, quote_mint @75, lp_mint @107, pool_base_token_account @139,
pool_quote_token_account @171, lp_supply u64 @203, coin_creator @211 (all zeros for pools made by hand;
pump.fun graduations carry the coin's creator). Hand-made pools put SOL on either side.

LP mint: a plain 82-byte Token-2022 mint whose mint authority is the pool itself. It changes only when liquidity
is added, removed or first minted, never on swaps, so a websocket on 82-byte Token-2022 accounts sees every new
pool (and every pull) for ~120 Helius credits a day."""
from __future__ import annotations

from tradingbot.data.helius import b58encode

PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
WSOL = "So11111111111111111111111111111111111111112"
NO_KEY = "11111111111111111111111111111111"  # 32 zero bytes in base58
LP_MINT_SIZE = 82


def parse_pool(raw: bytes) -> dict | None:
    if len(raw) < 243:
        return None
    k = lambda o: b58encode(raw[o:o + 32])
    base, quote = k(43), k(75)
    sol_is_base = base == WSOL
    if not sol_is_base and quote != WSOL:
        return None  # not a SOL pair
    return {
        "creator": k(11), "base_mint": base, "quote_mint": quote, "lp_mint": k(107),
        "base_vault": k(139), "quote_vault": k(171), "lp_supply": int.from_bytes(raw[203:211], "little"),
        "coin_creator": k(211), "handmade": k(211) == NO_KEY,
        "token_mint": quote if sol_is_base else base,
        "token_vault": k(171) if sol_is_base else k(139),
        "sol_vault": k(139) if sol_is_base else k(171),
    }


def parse_lp_mint(raw: bytes) -> dict | None:
    """Token-2022 base mint layout: COption<authority> (u32 tag + 32 bytes), supply u64, decimals u8."""
    if len(raw) != LP_MINT_SIZE:
        return None
    has_auth = int.from_bytes(raw[0:4], "little") == 1
    return {"authority": b58encode(raw[4:36]) if has_auth else None,
            "supply": int.from_bytes(raw[36:44], "little"), "decimals": raw[44]}


def token_amount(raw: bytes) -> int | None:
    """Amount held by an SPL / Token-2022 token account (u64 at byte 64)."""
    return int.from_bytes(raw[64:72], "little") if len(raw) >= 72 else None


def spot_price(sol_lamports: int, token_units: int, token_decimals: int) -> float | None:
    """SOL per whole token from the two vaults (what the backtest's chain price measures)."""
    if not sol_lamports or not token_units:
        return None
    return (sol_lamports / 1e9) / (token_units / 10 ** token_decimals)
