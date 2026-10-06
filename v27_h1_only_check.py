import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ЗАГРУЗКА ТОЛЬКО H1
# =============================================================================
def load_mt4_csv(path):
    sample = pd.read_csv(path, sep='\t', nrows=3, header=0)
    n_cols = sample.shape[1]
    if n_cols == 9:
        names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2']
    elif n_cols == 8:
        names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2']
    else:
        names = None
    df = pd.read_csv(path, sep='\t', header=0, names=names)
    if 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%Y.%m.%d %H:%M:%S')
    else:
        df['DateTime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
    return df

h1 = load_mt4_csv('EURUSD_H1_202401020000_202608262100.csv')

# =============================================================================
# 2. ТОЛЬКО H1 ПРИЗНАКИ
# =============================================================================
def add_features(data):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']
    out = pd.DataFrame(index=data.index)

    out['Return_1'] = c.pct_change(1)
    out['Return_4'] = c.pct_change(4)
    out['Dist_SMA20'] = (c - c.rolling(20).mean()) / c.rolling(20).mean()
    out['Dist_SMA50'] = (c - c.rolling(50).mean()) / c.rolling(50).mean()

    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    out['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    out['ATR_14'] = tr.rolling(14).mean()
    out['ATR_Ratio'] = out['ATR_14'] / (out['ATR_14'].rolling(50).mean() + 1e-10)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out['BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)

    out['MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)

    out['Hour'] = data.index.hour
    out['DayOfWeek'] = data.index.dayofweek
    out['London'] = out['Hour'].between(8, 15).astype(int)
    out['Overlap'] = out['Hour'].between(13, 15).astype(int)

    return out

df = add_features(h1)
df = df.iloc[250:].copy()

# =============================================================================
# 3. РЕЖИМЫ И TARGET
# =============================================================================
def regime_of(r):
    if (r['ATR_Ratio'] > 1.2) and (r['RSI_14'] > 60):
        return 'trend_hi'
    elif r['ATR_Ratio'] > 1.2:
        return 'range_hi'
    elif r['RSI_14'] > 60:
        return 'trend_lo'
    else:
        return 'range_lo'

df['Regime'] = df.apply(regime_of, axis=1)
regimes = ['trend_hi', 'range_hi', 'trend_lo', 'range_lo']

df['Target'] = (h1['Close'].reindex(df.index).shift(-1) > h1['Close'].reindex(df.index)).astype(int)
df.dropna(subset=['Target'], inplace=True)

exclude = ['Target', 'Regime']
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Данные: {len(df)} строк, признаков: {len(feature_cols)}")
print(f"Режимы:\n{df['Regime'].value_counts()}")

# =============================================================================
# 4. WALK-FORWARD
# =============================================================================
N_MODELS = 7
ALLOWED = ['range_lo', 'range_hi']
FIXED_THR = 0.60

test_months = pd.date_range('2025-03-01', '2026-07-01', freq='2MS')
pred_by_model = {s: pd.Series(dtype=float) for s in range(N_MODELS)}
edge = {m: {r: [] for r in regimes} for m in range(N_MODELS)}

for i, test_start in enumerate(test_months):
    test_end = test_start + pd.DateOffset(months=2)
    train = df[df.index < test_start]
    test = df[(df.index >= test_start) & (df.index < test_end)]

    if len(train) < 500 or len(test) < 30:
        continue

    X_tr, y_tr = train[feature_cols], train['Target']
    X_te = test[feature_cols]

    for seed in range(N_MODELS):
        m = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=100,
            reg_alpha=0.5, reg_lambda=2.0, random_state=seed * 7, verbose=-1
        )
        m.fit(X_tr, y_tr)
        prob = pd.Series(m.predict_proba(X_te)[:, 1], index=test.index)
        pred_by_model[seed] = pd.concat([pred_by_model[seed], prob])

        for r in regimes:
            sub = test[test['Regime'] == r]
            if len(sub) > 0:
                preds = (prob[sub.index] > 0.5).astype(int)
                correct = (preds.values == sub['Target'].values).sum()
                edge[seed][r].append(correct / len(sub))

    print(f"  WF {i+1}: {test_start.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}")

for m in range(N_MODELS):
    for r in regimes:
        edge[m][r] = np.mean(edge[m][r]) if edge[m][r] else 0.5

all_idx = sorted(set().union(*[set(pred_by_model[s].index) for s in range(N_MODELS)]))
final_prob = pd.Series(index=all_idx, dtype=float)
regime_at = {idx: df.loc[idx, 'Regime'] for idx in all_idx}

for idx in all_idx:
    r = regime_at[idx]
    ws, ps = [], []
    for s in range(N_MODELS):
        if idx in pred_by_model[s].index:
            e = edge[s][r]
            w = max(e - 0.5, 0.02)
            ws.append(w); ps.append(pred_by_model[s][idx])
    if not ws:
        final_prob[idx] = 0.5; continue
    ws = np.array(ws) / np.sum(ws)
    final_prob[idx] = np.dot(ws, ps)

final_prob.sort_index(inplace=True)
test_all = df.loc[final_prob.index].copy()
test_all['Prob'] = final_prob.values

# =============================================================================
# 5. БЭКТЕСТ
# =============================================================================
CAPITAL_START = 10000
SPREAD_PIPS = 0.3
COMMISSION_PER_LOT = 5
PIP_VALUE = 10
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0
MAX_RISK_PCT = 0.02
MAX_HOLD_BARS = 36
SWAP_PER_LOT_PER_HOUR = 0.03


def position_size_by_risk(capital, risk_pips):
    risk_dollars = capital * MAX_RISK_PCT
    lots = risk_dollars / (risk_pips * PIP_VALUE)
    return max(min(lots, 0.5), 0.01)


capital = CAPITAL_START
equity = []
trade_pnls = []
trade_meta = []
position = 0
entry_price = 0
entry_time = None
pos_size = 0
sl = 0
tp = 0
bars_in_trade = 0

for idx, row in test_all.iterrows():
    if position != 0:
        unrealized_pips = (h1.loc[idx, 'Close'] - entry_price) / 0.0001
        unrealized = unrealized_pips * PIP_VALUE * pos_size
        current_equity = capital + unrealized
    else:
        current_equity = capital
    equity.append(current_equity)

    if position != 0:
        bars_in_trade += 1
        high = h1.loc[idx, 'High']
        low = h1.loc[idx, 'Low']
        exit_reason = None
        if low <= sl:
            exit_pips = (sl - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'sl'
        elif high >= tp:
            exit_pips = (tp - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'tp'
        elif bars_in_trade >= MAX_HOLD_BARS:
            exit_pips = (h1.loc[idx, 'Close'] - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'time'
        else:
            continue
        swap = SWAP_PER_LOT_PER_HOUR * pos_size * bars_in_trade
        commission = COMMISSION_PER_LOT * pos_size
        pnl = exit_pips * PIP_VALUE * pos_size - commission - swap
        capital += pnl
        trade_pnls.append(pnl)
        trade_meta.append({'entry': entry_time, 'exit': idx, 'size': pos_size, 'pnl': pnl, 'reason': exit_reason, 'bars': bars_in_trade, 'swap': swap})
        position = 0; entry_price = 0; entry_time = None; pos_size = 0; sl = 0; tp = 0; bars_in_trade = 0

    if position == 0:
        if row['Regime'] not in ALLOWED: continue
        if row['Prob'] <= FIXED_THR: continue
        entry_price = h1.loc[idx, 'Close']
        entry_time = idx
        atr_pips = row['ATR_14'] / 0.0001
        risk_pips = atr_pips * ATR_SL_MULT + SPREAD_PIPS
        sl = entry_price - row['ATR_14'] * ATR_SL_MULT
        tp = entry_price + row['ATR_14'] * ATR_TP_MULT
        pos_size = position_size_by_risk(capital, risk_pips)
        position = 1; bars_in_trade = 0

if position != 0:
    last_idx = test_all.index[-1]
    last_close = h1.loc[last_idx, 'Close']
    exit_pips = (last_close - entry_price) / 0.0001 - SPREAD_PIPS
    swap = SWAP_PER_LOT_PER_HOUR * pos_size * bars_in_trade
    commission = COMMISSION_PER_LOT * pos_size
    pnl = exit_pips * PIP_VALUE * pos_size - commission - swap
    capital += pnl
    trade_pnls.append(pnl)
    trade_meta.append({'entry': entry_time, 'exit': last_idx, 'size': pos_size, 'pnl': pnl, 'reason': 'end', 'bars': bars_in_trade, 'swap': swap})
    equity[-1] = capital

equity = np.array(equity)
trade_pnls = np.array(trade_pnls)
n_trades = len(trade_pnls)

total_return = (equity[-1] - CAPITAL_START) / CAPITAL_START * 100
running_max = np.maximum.accumulate(equity)
drawdown = (equity / running_max - 1) * 100
max_dd = drawdown.min()
win_rate = (trade_pnls > 0).mean() if n_trades else 0
profit_factor = trade_pnls[trade_pnls > 0].sum() / abs(trade_pnls[trade_pnls < 0].sum()) if (trade_pnls < 0).any() and (trade_pnls > 0).any() else np.inf
sharpe = trade_pnls.mean() / (trade_pnls.std() + 1e-10) * np.sqrt(252) if n_trades else 0
calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else np.inf
avg_bars = np.mean([t['bars'] for t in trade_meta]) if trade_meta else 0
total_swap = sum(t['swap'] for t in trade_meta) if trade_meta else 0

print(f"\n{'='*60}")
print(f"РЕЗУЛЬТАТЫ БЭКТЕСТА v27 (H1 Only Check)")
print(f"{'='*60}")
print(f"  Сделок: {n_trades} | WR: {win_rate:.1%} | Ret: {total_return:+.2f}%")
print(f"  MaxDD: {max_dd:.2f}% | Sharpe: {sharpe:.2f} | PF: {profit_factor:.2f}")
print(f"  Final: ${equity[-1]:,.2f}")
print(f"{'='*60}")

# Сохранение
pd.DataFrame({'Equity': equity, 'Drawdown_%': drawdown}, index=test_all.index).to_csv('equity_v27_h1_only.csv')
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eq := pd.Series(equity, index=test_all.index), color='teal')
ax.axhline(y=CAPITAL_START, color='gray', linestyle='--')
ax.set_title(f'v27 H1 Only | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f}')
plt.tight_layout()
plt.savefig('backtest_v27_h1_only.png', dpi=150)
plt.close()
print("Сохранено: equity_v27_h1_only.csv, backtest_v27_h1_only.png")
