# MacroGuardian V6 — AI Agent API

An AI-native trading signal service powered by the MacroGuardian V6 quantitative engine.

## Proprietary Indicators

| Indicator | Full Name | Scale | Description |
|-----------|-----------|-------|-------------|
| **BCI** | Bull/Bear Climate Index | 700–1300 | Macro trend filter (above 1000 = bull) |
| **MCO** | Momentum Cycle Oscillator | -1.0 to +1.0 | Weekly momentum signal |
| **SOI** | Sentiment Overheat Indicator | 0 or 1 | Overheat risk detection |
| **VG** | Volatility Guardrail | Boolean | Black swan circuit breaker |

## API Endpoints

| Endpoint | Access | Description |
|----------|--------|-------------|
| `GET /` | Free | Service info |
| `GET /performance` | Free | Strategy performance & methodology |
| `GET /signal/{symbol}` | Paid | Real-time trading signal |
| `GET /history/{symbol}` | Paid | Historical indicator data |
| `GET /docs` | Free | Swagger UI documentation |

## Deployment

Deployed on Railway with GitHub auto-deploy.

### Environment Variables (set in Railway)

- `MG_OWNER_API_KEY` — Owner master API key
- `MG_DEMO_API_KEY` — Demo/public API key  
- `MG_RECEIVING_WALLET` — USDC payment wallet address

## Security Features

1. **Multi-endpoint failover** — 6 Binance API backup endpoints with auto-switching
2. **PSN transformation** — Proprietary Scale Normalization hides original indicator logic
3. **Environment variable encapsulation** — All sensitive keys stored as env vars
