"""
MacroGuardian V6 — AI Agent API (Code B)
=========================================
This is the externally-facing API service. It wraps the core trading engine
(Code A) and exposes its analysis through branded, IP-protected endpoints.

Key Features:
- Proprietary terminology (BCI, MCO, SOI, STV, MTT, VG)
- Proprietary Scale Normalization (PSN) on all indicator outputs
- Proxy Payment Gateway (API Key based, upgradeable to x402 in the future)
- /signal, /performance, /history endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

# ============================================================
# [SECTION 1] Configuration (Environment Variable Encapsulation)
# ============================================================

TZ_OFFSET = timezone(timedelta(hours=8))

# Proxy Payment Gateway: API Key whitelist
# Sensitive keys loaded from Railway Environment Variables for security.
# Fallback defaults are provided for local development only.
_owner_key = os.environ.get("MG_OWNER_API_KEY", "mg-owner-master-key-2026")
_demo_key = os.environ.get("MG_DEMO_API_KEY", "mg-demo-key-public")
VALID_API_KEYS = {
    _owner_key: "owner",
    _demo_key: "demo",
}

# Receiving wallet for future x402 integration
# Loaded from Railway Environment Variables for security.
RECEIVING_WALLET = os.environ.get(
    "MG_RECEIVING_WALLET",
    "0xFf44255854D9a42Ff296F9a6620272adDC98DdD2"
)

# ============================================================
# [SECTION 2] Proprietary Scale Normalization (PSN) Functions
# ============================================================

def psn_bci(raw_value: float) -> float:
    """BCI (Bull/Bear Climate Index): raw 0-100 -> branded 700-1300"""
    return round(700 + (raw_value * 6), 1)

def psn_mco(raw_value: float) -> float:
    """MCO (Momentum Cycle Oscillator): raw 0-100 -> branded -1.0 to +1.0"""
    return round((raw_value / 50) - 1, 4)

def psn_soi(is_active: bool) -> int:
    """SOI (Sentiment Overheat Indicator): boolean -> 0 or 1"""
    return 1 if is_active else 0

# ============================================================
# [SECTION 3] Core Engine (from Code A - MacroGuardian_v2026_1_final.py)
# ============================================================

# Binance API endpoints (multiple fallbacks for cloud server compatibility)
BINANCE_ENDPOINTS = [
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api.binance.com",
    "https://data-api.binance.vision",
]

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

import time as _time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MacroGuardian")


def fetch_data(symbol, interval, limit=300):
    """Fetch kline data with multi-endpoint fallback and retry logic."""
    for endpoint in BINANCE_ENDPOINTS:
        for attempt in range(2):
            try:
                url = f"{endpoint}/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
                logger.info(f"Fetching {symbol} {interval} from {endpoint} (attempt {attempt+1})")
                r = requests.get(url, timeout=15, headers=REQUEST_HEADERS)
                if r.status_code == 451:  # Geo-restricted
                    logger.warning(f"{endpoint} returned 451 (geo-restricted), trying next...")
                    break
                if r.status_code == 403:
                    logger.warning(f"{endpoint} returned 403 (forbidden), trying next...")
                    break
                if r.status_code == 429:  # Rate limited
                    logger.warning(f"{endpoint} returned 429 (rate limited), waiting...")
                    _time.sleep(2)
                    continue
                if r.status_code != 200:
                    logger.warning(f"{endpoint} returned {r.status_code}, trying next...")
                    break
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    logger.warning(f"{endpoint} returned empty data for {symbol}")
                    break
                df = pd.DataFrame(data, columns=[
                    't', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 'nt', 'tb', 'tq', 'i'
                ])
                df[["o", "h", "l", "c"]] = df[["o", "h", "l", "c"]].astype(float)
                df["time"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize('UTC').dt.tz_convert(TZ_OFFSET)
                logger.info(f"Successfully fetched {len(df)} rows for {symbol} {interval} from {endpoint}")
                return df
            except requests.exceptions.Timeout:
                logger.warning(f"{endpoint} timed out (attempt {attempt+1})")
                continue
            except Exception as e:
                logger.error(f"{endpoint} error: {str(e)}")
                break
    logger.error(f"All endpoints failed for {symbol} {interval}")
    return None


def fetch_realtime_price(symbol):
    """Fetch realtime price with multi-endpoint fallback."""
    for endpoint in BINANCE_ENDPOINTS:
        try:
            url = f"{endpoint}/api/v3/ticker/price?symbol={symbol}USDT"
            r = requests.get(url, timeout=10, headers=REQUEST_HEADERS)
            if r.status_code != 200:
                continue
            data = r.json()
            price = float(data.get("price", 0))
            if price > 0:
                return price
        except Exception:
            continue
    return 0.0

def calc_skdj(df, period):
    if df is None or len(df) < period:
        return df
    ln = df["l"].rolling(window=period).min()
    hn = df["h"].rolling(window=period).max()
    df[f"RSV{period}"] = (df["c"] - ln) / (hn - ln + 1e-9) * 100
    rsv = df[f"RSV{period}"].fillna(50).tolist()
    k, d = [50.0], [50.0]
    for i in range(1, len(rsv)):
        nk = (rsv[i] + 2 * k[-1]) / 3
        nd = (nk + 2 * d[-1]) / 3
        k.append(nk)
        d.append(nd)
    df[f"K{period}"] = k
    df[f"D{period}"] = d
    df[f"DK{period}"] = df[f"K{period}"].diff()
    df[f"DD{period}"] = df[f"D{period}"].diff()
    df[f"DIFF{period}"] = df[f"K{period}"] - df[f"D{period}"]
    return df

def calc_ma(df, col, period):
    df[f"MA{period}"] = df[col].rolling(period).mean()
    df[f"DMA{period}"] = df[f"MA{period}"].diff()
    return df

class MacroEngine:
    def __init__(self):
        self.d8_peak = 0.0
        self.current_weight = 0.0
        self.is_reduced = False
        self.residual_weight = 0.0
        self.prev_diff8_positive = None
        self.hlf_blocked_weeks = 0
        self.hlf_blocked_price = 0.0
        self.prev_week_high = 0.0
        self.prev_prev_week_high = 0.0

    def _should_hlf_block(self, l, price):
        if self.d8_peak <= 65:
            return False, ""
        ma8_val = l["MA8"] if not pd.isna(l["MA8"]) else 0
        dma8_val = l["DMA8"] if not pd.isna(l["DMA8"]) else 0
        if price > ma8_val and ma8_val > 0 and dma8_val > 0:
            self.hlf_blocked_weeks = 0
            self.hlf_blocked_price = 0
            return False, "SOI cleared: price above rising STV"
        if self.d8_peak > 85:
            dynamic_threshold = 55
        elif self.d8_peak > 75:
            dynamic_threshold = 60
        else:
            dynamic_threshold = 65
        current_d8 = l["D8"]
        if current_d8 < dynamic_threshold:
            self.hlf_blocked_weeks = 0
            self.hlf_blocked_price = 0
            return False, f"SOI cleared: MCO_D below dynamic threshold"
        if self.hlf_blocked_weeks >= 2:
            if (l["h"] > self.prev_week_high > self.prev_prev_week_high
                    and self.prev_week_high > 0 and self.prev_prev_week_high > 0):
                self.hlf_blocked_weeks = 0
                self.hlf_blocked_price = 0
                return False, "SOI exemption: consecutive breakout highs"
        return True, f"SOI active (overheat detected)"

    def analyze(self, sym):
        dw = fetch_data(sym, "1w", 100)
        dm = fetch_data(sym, "1M", 50)
        if dw is None or dm is None or len(dw) < 20:
            return None
        dw = calc_skdj(dw, 8)
        dw = calc_ma(dw, "c", 8)
        dw = calc_ma(dw, "c", 20)
        dm = calc_skdj(dm, 3)
        dm = calc_ma(dm, "c", 3)
        l = dw.iloc[-1]
        p = dw.iloc[-2]
        ml = dm.iloc[-1]
        mp = dm.iloc[-2]
        diag = ""
        hlf_blocked = False

        # D8_peak update
        curr_diff8_positive = (l["DIFF8"] > 0)
        if self.prev_diff8_positive is not None and self.prev_diff8_positive and not curr_diff8_positive:
            self.d8_peak = l["D8"]
        self.prev_diff8_positive = curr_diff8_positive

        # Macro filter
        m_safe = (ml["DIFF3"] > 0 and ml["DK3"] > 0 and ml["DD3"] > 0
                  and ml["c"] > ml["MA3"] and ml["DMA3"] > 0)
        if not m_safe:
            self.current_weight = 0.0
            self.is_reduced = False
            self.residual_weight = 0.0
            diag = "BCI blocked"
            return self._build_result(sym, diag, l, p, ml, mp, m_safe, hlf_blocked)

        # Death cross
        if l["DIFF8"] <= 0 and l["DK8"] <= 0 and l["DD8"] <= 0:
            self.current_weight = 0.0
            self.is_reduced = False
            self.residual_weight = 0.0
            diag = "MCO death cross - full exit"
            return self._build_result(sym, diag, l, p, ml, mp, m_safe, hlf_blocked)

        # HLF / SOI check
        kd8_gold = (l["DIFF8"] > 0 and l["DK8"] > 0 and l["DD8"] > 0)
        if kd8_gold:
            should_block, block_reason = self._should_hlf_block(l, l["c"])
            if should_block:
                self.hlf_blocked_weeks += 1
                if self.hlf_blocked_price == 0:
                    self.hlf_blocked_price = l["c"]
                hlf_blocked = True
                diag = block_reason
                return self._build_result(sym, diag, l, p, ml, mp, m_safe, hlf_blocked)
            else:
                self.hlf_blocked_weeks = 0
                self.hlf_blocked_price = 0

        # Position sizing
        target = self.current_weight
        if kd8_gold:
            if self.is_reduced:
                if l["c"] > l["MA8"]:
                    new_weight = min(self.residual_weight + 0.30, 1.0)
                    target = new_weight
                    self.is_reduced = False
                    diag = f"Re-entry ({target*100:.0f}%)"
                    if l["c"] > l["MA20"] and l["DMA20"] > 0 and target < 1.0:
                        target = 1.0
                        diag = "Re-entry -> Full position"
                    elif l["c"] > l["MA8"] and l["DMA8"] > 0 and target < 0.8:
                        target = 0.8
                        diag = "Re-entry -> 80% position"
                else:
                    diag = f"Holding residual ({self.current_weight*100:.0f}%)"
            else:
                if l["c"] > l["MA20"] and l["DMA20"] > 0:
                    target = 1.0
                    diag = "Full position (100%)"
                elif l["c"] > l["MA8"] and l["DMA8"] > 0:
                    target = 0.8
                    diag = "Add position (80%)"
                else:
                    target = 0.3
                    diag = "Trial position (30%)"
        else:
            diag = "Awaiting MCO golden cross"
        self.current_weight = target

        # Breakdown reduction
        ma8_val = l["MA8"] if not pd.isna(l["MA8"]) else 0
        if l["c"] < ma8_val and self.current_weight > 0 and not self.is_reduced:
            self.residual_weight = self.current_weight * 0.5
            self.current_weight = self.residual_weight
            self.is_reduced = True
            diag = f"Breakdown reduction ({self.current_weight*100:.0f}%)"

        # Black swan / VG
        realtime_price = fetch_realtime_price(sym)
        if (ma8_val > 0 and self.current_weight > 0 and realtime_price > 0
                and (ma8_val - realtime_price) >= (0.15 * ma8_val)):
            self.current_weight = 0.0
            self.is_reduced = False
            self.residual_weight = 0.0
            diag = "VG triggered - emergency exit"

        self.prev_prev_week_high = self.prev_week_high
        self.prev_week_high = l["h"]
        return self._build_result(sym, diag, l, p, ml, mp, m_safe, hlf_blocked)

    def _build_result(self, sym, diag, l, p, ml, mp, m_safe, hlf_blocked):
        """Build result with PSN-transformed branded indicators"""
        return {
            "symbol": sym,
            "diagnosis": diag,
            "recommended_weight": float(self.current_weight),
            "price": float(l["c"]),
            "change_1w_pct": float((l["c"] / p["c"] - 1) * 100),
            # BCI (Bull/Bear Climate Index) - PSN transformed
            "bci_k": psn_bci(float(ml["K3"])),
            "bci_d": psn_bci(float(ml["D3"])),
            "bci_safe": bool(m_safe),
            # MCO (Momentum Cycle Oscillator) - PSN transformed
            "mco_k": psn_mco(float(l["K8"])),
            "mco_d": psn_mco(float(l["D8"])),
            # SOI (Sentiment Overheat Indicator) - PSN transformed
            "soi_status": psn_soi(hlf_blocked),
            # VG (Volatility Guardrail) status
            "vg_active": bool("VG triggered" in diag),
        }


# ============================================================
# [SECTION 4] Proxy Payment Gateway (API Key Middleware)
# ============================================================

async def verify_api_key(x_api_key: str = Header(None)):
    """
    Proxy Payment Gateway: Validates API keys for paid endpoints.
    Returns 402 Payment Required if no valid key is provided.
    This will be replaced by x402 native payment in a future update.
    """
    if x_api_key is None or x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Payment Required",
                "message": "A valid API key is required to access this endpoint.",
                "pricing": {
                    "signal_per_call": "$0.10 USD",
                    "history_per_call": "$0.05 USD",
                    "monthly_unlimited": "$29.99 USD",
                },
                "payment_methods": [
                    {
                        "method": "USDC on Base",
                        "wallet": RECEIVING_WALLET,
                        "note": "Send payment and contact support for API key issuance."
                    },
                ],
                "contact": "Contact the MacroGuardian team for API key registration.",
                "x402_note": "Native x402 protocol payment coming soon."
            }
        )
    return VALID_API_KEYS[x_api_key]


# ============================================================
# [SECTION 5] FastAPI Application & Endpoints
# ============================================================

app = FastAPI(
    title="MacroGuardian V6 Signal API",
    description=(
        "An AI-native trading signal service powered by the MacroGuardian V6 engine. "
        "Provides real-time analysis using proprietary indicators including "
        "BCI (Bull/Bear Climate Index), MCO (Momentum Cycle Oscillator), "
        "SOI (Sentiment Overheat Indicator), and VG (Volatility Guardrail). "
        "Validated across 50 cryptocurrencies with 92% outperformance vs buy-and-hold."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine instances (one per symbol to maintain state)
engine_map = {}


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "MacroGuardian V6 Signal API",
        "version": "1.0.0",
        "endpoints": {
            "/performance": "Free - Strategy performance and methodology (no key required)",
            "/signal/{symbol}": "Paid - Real-time trading signal (API key required)",
            "/history/{symbol}": "Paid - Historical indicator data for charting (API key required)",
            "/docs": "Interactive API documentation",
        },
        "payment_wallet": RECEIVING_WALLET,
    }


@app.get("/performance", tags=["Performance (Free)"])
async def get_performance():
    """
    Free endpoint. Provides a comprehensive summary of the MacroGuardian V6
    strategy's historical performance, methodology, and proprietary indicators.
    Designed for due diligence by AI agents or human investors.
    """
    return {
        "strategy_name": "MacroGuardian V6",
        "philosophy": "Risk-first trend following with built-in Volatility Guardrail (VG) for black swan protection.",
        "proprietary_indicators": {
            "BCI": {
                "name": "Bull/Bear Climate Index",
                "description": "A macro indicator that determines the market's primary trend (bull or bear climate).",
                "scale": {
                    "range": "700 to 1300",
                    "interpretation": {
                        "above_1000": "Strong Bull Market Climate - system authorized for full positions",
                        "850_to_1000": "Moderate Climate - system authorized with caution",
                        "below_850": "Bear Market Climate - all positions prohibited, capital preserved"
                    }
                }
            },
            "MCO": {
                "name": "Momentum Cycle Oscillator",
                "description": "A weekly indicator capturing mid-term momentum and cyclical movements.",
                "scale": {
                    "range": "-1.0 to +1.0",
                    "interpretation": {
                        "above_0.5": "Strong bullish momentum",
                        "0_to_0.5": "Moderate bullish momentum",
                        "negative_0.5_to_0": "Weakening momentum, caution advised",
                        "below_negative_0.5": "Strong bearish momentum, risk-off"
                    }
                }
            },
            "SOI": {
                "name": "Sentiment Overheat Indicator",
                "description": "A risk management module that identifies periods of irrational exuberance to prevent buying at market tops.",
                "scale": {
                    "range": "0 or 1",
                    "interpretation": {
                        "0": "Normal - no overheat detected",
                        "1": "Overheat active - new entries blocked until cooldown"
                    }
                }
            },
            "VG": {
                "name": "Volatility Guardrail",
                "description": "An automated circuit breaker that triggers emergency exit of all positions during sudden, extreme market crashes.",
                "trigger": "Activates when price deviation from trend exceeds critical threshold"
            }
        },
        "backtest_period": "2017-01-01 to 2026-02-25",
        "top_50_validation": {
            "symbols_tested": 50,
            "symbols_outperformed_buy_and_hold": 46,
            "win_rate": "92%",
            "average_strategy_roi": "+1653.3%",
            "average_buy_and_hold_roi": "+339.1%",
            "average_max_drawdown_strategy": "-34.4%",
            "average_max_drawdown_buy_and_hold": "-91.9%",
            "risk_reduction": "57.5%"
        },
        "black_swan_protection_record": [
            {"event": "COVID-19 Crash (2020-03)", "result": "Avoided - VG triggered, 0% loss"},
            {"event": "China Mining Ban (2021-05)", "result": "Avoided - BCI signaled bear climate"},
            {"event": "LUNA/UST Collapse (2022-05)", "result": "Avoided - BCI signaled bear climate"},
            {"event": "FTX Collapse (2022-11)", "result": "Avoided - BCI signaled bear climate"},
            {"event": "SVB Banking Crisis (2023-03)", "result": "Avoided - VG triggered"},
            {"event": "Japan Carry Trade Unwind (2024-08)", "result": "Avoided - VG triggered"},
            {"event": "DeepSeek AI Crash (2025-01)", "result": "Avoided - VG triggered"}
        ],
        "api_pricing": {
            "signal_per_call": "$0.10 USD",
            "history_per_call": "$0.05 USD",
            "monthly_unlimited": "$29.99 USD",
            "payment_wallet_usdc_base": RECEIVING_WALLET
        }
    }


@app.get("/signal/{symbol}", tags=["Trading Signals (Paid)"])
async def get_signal(symbol: str, role: str = Depends(verify_api_key)):
    """
    Paid endpoint. Analyzes the specified cryptocurrency symbol and returns
    the latest trading signal with all proprietary indicators (PSN-transformed).
    Requires a valid API key in the X-API-Key header.
    """
    sym_upper = symbol.upper()
    if sym_upper not in engine_map:
        engine_map[sym_upper] = MacroEngine()
    engine = engine_map[sym_upper]
    try:
        result = engine.analyze(sym_upper)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch data for symbol '{sym_upper}'. Please verify it is a valid Binance USDT trading pair."
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {str(e)}")


@app.get("/history/{symbol}", tags=["Historical Data (Paid)"])
async def get_history(symbol: str, weeks: int = 52, role: str = Depends(verify_api_key)):
    """
    Paid endpoint. Returns historical weekly indicator data for the specified symbol,
    with all values PSN-transformed. Ideal for charting and visualization.
    Requires a valid API key in the X-API-Key header.

    - **symbol**: Cryptocurrency symbol (e.g., BTC, ETH, SOL)
    - **weeks**: Number of weeks of history to return (default: 52, max: 200)
    """
    sym_upper = symbol.upper()
    weeks = min(weeks, 200)

    dw = fetch_data(sym_upper, "1w", weeks + 20)
    dm = fetch_data(sym_upper, "1M", 50)

    if dw is None or dm is None or len(dw) < 20:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch data for symbol '{sym_upper}'."
        )

    dw = calc_skdj(dw, 8)
    dw = calc_ma(dw, "c", 8)
    dw = calc_ma(dw, "c", 20)
    dm = calc_skdj(dm, 3)
    dm = calc_ma(dm, "c", 3)

    # Build monthly BCI lookup
    bci_lookup = {}
    for i in range(1, len(dm)):
        row = dm.iloc[i]
        month_key = row["time"].strftime("%Y-%m")
        m_safe = (row["DIFF3"] > 0 and row["DK3"] > 0 and row["DD3"] > 0
                  and row["c"] > row["MA3"] and row["DMA3"] > 0)
        bci_lookup[month_key] = {
            "bci_k": psn_bci(float(row["K3"])),
            "bci_d": psn_bci(float(row["D3"])),
            "bci_safe": bool(m_safe),
        }

    # Build weekly history with PSN
    history = []
    tail = dw.tail(weeks)
    for _, row in tail.iterrows():
        if pd.isna(row.get("K8")) or pd.isna(row.get("D8")):
            continue
        month_key = row["time"].strftime("%Y-%m")
        bci_data = bci_lookup.get(month_key, {"bci_k": 1000.0, "bci_d": 1000.0, "bci_safe": False})
        history.append({
            "date": row["time"].strftime("%Y-%m-%d"),
            "price": float(row["c"]),
            # MCO (PSN transformed)
            "mco_k": psn_mco(float(row["K8"])),
            "mco_d": psn_mco(float(row["D8"])),
            # BCI (PSN transformed, from monthly data)
            "bci_k": bci_data["bci_k"],
            "bci_d": bci_data["bci_d"],
            "bci_safe": bci_data["bci_safe"],
            # STV / MTT
            "stv": float(row["MA8"]) if not pd.isna(row["MA8"]) else None,
            "mtt": float(row["MA20"]) if not pd.isna(row["MA20"]) else None,
        })

    return {
        "symbol": sym_upper,
        "period": f"{weeks} weeks",
        "data_points": len(history),
        "indicator_scales": {
            "bci": "700 to 1300 (above 1000 = bull climate)",
            "mco": "-1.0 to +1.0 (above 0 = bullish momentum)",
            "stv_mtt": "Absolute price levels (trend reference lines)",
        },
        "history": history,
    }
