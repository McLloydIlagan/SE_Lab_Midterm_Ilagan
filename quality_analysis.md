# Software Quality Analysis Report - ISO/IEC 25010

## Project: Cryptocurrency Analysis Tool | Repository: SE_LabMidterm_Ilagan | Date: April 2026

---

## 1. Introduction

This document analyzes the software quality of my Cryptocurrency Analysis Tool according to the ISO/IEC 25010 quality model. The tool fetches real-time cryptocurrency data from CoinGecko API, calculates technical indicators (RSI, MACD, Fibonacci), predicts future prices using Random Forest machine learning, and backtests trading strategies.

---

## 2. ISO/IEC 25010 Quality Attributes

### Quality Attribute 1: Reliability - Fault Tolerance

**Definition:** The ability of the system to maintain its level of performance even when facing software faults or invalid inputs.

**How my module implements this:**

| Mechanism | Implementation | Code Example |
| Input Validation | All public methods validate inputs | `if not symbol: raise ValueError` |
| Error Handling | Try-catch blocks for API calls | `try: requests.get() except: raise Exception` |
| Data Validation | Check minimum data requirements | `if len(df) < 14: raise Exception` |
| Timeout Protection | API requests have timeouts | `requests.get(timeout=10)` |

**Code Example from my module:**
```python
def fetch_ohlcv(self, symbol: str, days: int = 365):
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    if days < 1 or days > 365:
        raise ValueError("Days must be between 1 and 365")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch data: {str(e)}")
