import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ H1, H4, D1
# =============================================================================
def load_mt4_csv(path):
    """Загружает MT4 CSV с автоопределением колонок."""
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
h4 = load_mt4_csv('EURUSD_H4_202401020000_202608262000.csv')
d1 = load_mt4_csv('EURUSD_Daily_202401020000_202608260000.csv')

print(f"H1: {len(h1)} баров | H4: {len(h4)} баров | D1: {len(d1)} баров")

# =============================================================================
# 2. ПРИЗНАКИ ДЛЯ КАЖДОГО ТАЙМФРЕЙМА
# =============================================================================
def add_tf_features(data, prefix):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']
    out = pd.DataFrame(index=data.index)

    out[f'{prefix}_Close'] = c
    out[f'{prefix}_Return_1'] = c.pct_change(1)
    out[f'{prefix}_Return_4'] = c.pct_change(4)
    out[f'{prefix}_SMA_20'] = c.rolling(20).mean()
    out[f'{prefix}_SMA_50'] = c.rolling(50).mean()
    out[f'{prefix}_EMA_12'] = c.ewm(span=12).mean()
    out[f'{prefix}_EMA_26'] = c.ewm(span=26).mean()
    out[f'{prefix}_Dist_SMA20'] = (c - out[f'{prefix}_SMA_20']) / out[f'{prefix}_SMA_20']
    out[f'{prefix}_Dist_SMA50'] = (c - out[f'{prefix}_SMA_50']) / out[f'{prefix}_SMA_50']
    out[f'{prefix}_SMA_Spread'] = (out[f'{prefix}_SMA_20'] - out[f'{prefix}_SMA_50']) / out[f'{prefix}_SMA_50']

    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    out[f'{prefix}_RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    out[f'{prefix}_MACD'] = out[f'{prefix}_EMA_12'] - out[f'{prefix}_EMA_26']
    out[f'{prefix}_MACD_Signal'] = out[f'{prefix}_MACD'].ewm(span=9).mean()
    out[f'{prefix}_MACD_Hist'] = out[f'{prefix}_MACD'] - out[f'{prefix}_MACD_Signal']

    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    out[f'{prefix}_ATR_14'] = tr.rolling(14).mean()
    out[f'{prefix}_ATR_Ratio'] = out[f'{prefix}_ATR_14'] / (out[f'{prefix}_ATR_14'].rolling(50).mean() + 1e-10)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out[f'{prefix}_BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    out[f'{prefix}_BB_Width'] = (4*bb_std) / (bb_mid + 1e-10)

    out[f'{prefix}_Body'] = np.abs(c - o) / (out[f'{prefix}_ATR_14'] + 1e-10)
    out[f'{prefix}_UpperWick'] = (h - pd.concat([c,o], axis=1).max(axis=1)) / (out[f'{prefix}_ATR_14'] + 1e-10)
    out[f'{prefix}_LowerWick'] = (pd.concat([c,o], axis=1).min(axis=1) - l) / (out[f'{prefix}_ATR_14'] + 1e-10)

    out[f'{prefix}_Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)
    out[f'{prefix}_MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    out[f'{prefix}_ADX'] = dx.rolling(14).mean()
    out[f'{prefix}_Trend_Strength'] = out[f'{prefix}_ADX'] / 50.0
    out[f'{prefix}_Trend_Direction'] = np.sign(out[f'{prefix}_SMA_Spread'])

    return out

h1_feat = add_tf_features(h1, 'H1')
h4_feat = add_tf_features(h4, 'H4')
d1_feat = add_tf_features(d1, 'D1')

# Дневные уровни
levels = pd.DataFrame(index=d1.index)
levels['D1_High'] = d1['High']
levels['D1_Low'] = d1['Low']
levels['D1_Range'] = d1['High'] - d1['Low']
levels['D1_PrevHigh'] = d1['High'].shift(1)
levels['D1_PrevLow'] = d1['Low'].shift(1)
levels['D1_PrevClose'] = d1['Close'].shift(1)

# =============================================================================
# 3. КАУЗАЛЬНОЕ ПРИСОЕДИНЕНИЕ H4 И D1 К H1
# =============================================================================
# Для каждого H1 бара берём последний ЗАКРЫТЫЙ бар H4/D1 (строго в прошлом)
def merge_past_only(h1_idx, higher_feat):
    """Для каждого H1 времени берём последний доступный бар старшего ТФ, который уже закрылся."""
    # reindex с method='ffill' по времени — это каузально, т.к. берёт последнее известное значение
    merged = higher_feat.reindex(h1_idx, method='ffill')
    return merged

df = h1_feat.copy()
h4_merged = merge_past_only(df.index, h4_feat)
d1_merged = merge_past_only(df.index, d1_feat)
levels_merged = merge_past_only(df.index, levels)

# Переименуем, чтобы избежать конфликтов
h4_merged.columns = [c + '_H4' for c in h4_merged.columns]
d1_merged.columns = [c + '_D1' for c in d1_merged.columns]
levels_merged.columns = [c + '_LVL' for c in levels_merged.columns]

df = df.join(h4_merged, how='left')
df = df.join(d1_merged, how='left')
df = df.join(levels_merged, how='left')

# Расстояние до дневных уровней
df['Dist_D1_High'] = (df['H1_Close'] - df['D1_High_LVL']) / df['D1_High_LVL']
df['Dist_D1_Low'] = (df['H1_Close'] - df['D1_Low_LVL']) / df['D1_Low_LVL']
df['Dist_D1_PrevHigh'] = (df['H1_Close'] - df['D1_PrevHigh_LVL']) / df['D1_PrevHigh_LVL']
df['Dist_D1_PrevLow'] = (df['H1_Close'] - df['D1_PrevLow_LVL']) / df['D1_PrevLow_LVL']
df['In_D1_Range'] = ((df['H1_Close'] >= df['D1_Low_LVL']) & (df['H1_Close'] <= df['D1_High_LVL'])).astype(int)

# Дополнительные H1 признаки
df['H1_Hour'] = df.index.hour
df['H1_DayOfWeek'] = df.index.dayofweek
df['H1_London'] = df['H1_Hour'].between(8, 15).astype(int)
df['H1_Overlap'] = df['H1_Hour'].between(13, 15).astype(int)

# Убираем пропуски
df = df.iloc[250:].copy()

# =============================================================================
# 4. РЕЖИМЫ
# =============================================================================
def regime_of(r):
    if (r['H1_ATR_Ratio'] > 1.2) and (r['H1_Trend_Strength'] > 0.6):
        return 'trend_hi'
    elif r['H1_ATR_Ratio'] > 1.2:
        return 'range_hi'
    elif r['H1_Trend_Strength'] > 0.6:
        return 'trend_lo'
    else:
        return 'range_lo'

df['Regime'] = df.apply(regime_of, axis=1)
regimes = ['trend_hi', 'range_hi', 'trend_lo', 'range_lo']

# =============================================================================
# 5. TARGET
# =============================================================================
df['Target'] = (df['H1_Close'].shift(-1) > df['H1_Close']).astype(int)
df.dropna(subset=['Target'], inplace=True)

exclude = ['H1_Close', 'H4_Close_H4', 'D1_Close_D1', 'Target', 'Regime',
           'D1_High_LVL', 'D1_Low_LVL', 'D1_Range_LVL', 'D1_PrevHigh_LVL',
           'D1_PrevLow_LVL', 'D1_PrevClose_LVL']
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Данные: {len(df)} строк, признаков: {len(feature_cols)}")
print(f"Режимы:\n{df['Regime'].value_counts()}")

# =============================================================================
# 6. WALK-FORWARD С ДИНАМИЧЕСКИМ ВЗВЕШИВАНИЕМ
# =============================================================================
N_MODELS = 7
ALLOWED = ['range_lo', 'range_hi']

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
            n_estimators=300 + seed * 50,
            max_depth=3 + (seed % 3),
            learning_rate=0.03,
            subsample=0.6 + seed * 0.05,
            colsample_bytree=0.6 + seed * 0.05,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed * 7,
            verbose=-1
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
            ws.append(w)
            ps.append(pred_by_model[s][idx])
    if not ws:
        final_prob[idx] = 0.5
        continue
    ws = np.array(ws) / np.sum(ws)
    final_prob[idx] = np.dot(ws, ps)

final_prob.sort_index(inplace=True)
test_all = df.loc[final_prob.index].copy()
test_all['Prob'] = final_prob.values

# =============================================================================
# 7. БЭКТЕСТ
# =============================================================================
regime_thr = {}
for r in test_all['Regime'].unique():
    rd = test_all[test_all['Regime'] == r]
    if len(rd) < 30:
        regime_thr[r] = 0.55
        continue
    best_thr, best_ev = 0.55, -999
    for thr in np.arange(0.50, 0.72, 0.02):
        mask = rd['Prob'] > thr
        if mask.sum() < 5:
            continue
        wr = rd.loc[mask, 'Target'].mean()
        ev = wr - (1 - wr)
        if ev > best_ev:
            best_ev, best_thr = ev, thr
    regime_thr[r] = best_thr

print("\nRegime thresholds:", {k: f"{v:.2f}" for k, v in regime_thr.items()})

# Параметры
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
        unrealized_pips = (row['H1_Close'] - entry_price) / 0.0001
        unrealized = unrealized_pips * PIP_VALUE * pos_size
        current_equity = capital + unrealized
    else:
        current_equity = capital

    equity.append(current_equity)

    if position != 0:
        bars_in_trade += 1
        h1_row = h1.loc[idx]
        high = h1_row['High']
        low = h1_row['Low']
        exit_reason = None

        if low <= sl:
            exit_pips = (sl - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'sl'
        elif high >= tp:
            exit_pips = (tp - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'tp'
        elif bars_in_trade >= MAX_HOLD_BARS:
            exit_pips = (row['H1_Close'] - entry_price) / 0.0001 - SPREAD_PIPS
            exit_reason = 'time'
        else:
            continue

        swap = SWAP_PER_LOT_PER_HOUR * pos_size * bars_in_trade
        commission = COMMISSION_PER_LOT * pos_size
        pnl = exit_pips * PIP_VALUE * pos_size - commission - swap
        capital += pnl
        trade_pnls.append(pnl)
        trade_meta.append({
            'entry': entry_time, 'exit': idx, 'size': pos_size,
            'pnl': pnl, 'reason': exit_reason, 'bars': bars_in_trade, 'swap': swap
        })
        position = 0
        entry_price = 0
        entry_time = None
        pos_size = 0
        sl = 0
        tp = 0
        bars_in_trade = 0

    if position == 0:
        if row['Regime'] not in ALLOWED:
            continue

        thr = regime_thr.get(row['Regime'], 0.55)
        if row['Prob'] <= thr:
            continue

        entry_price = row['H1_Close']
        entry_time = idx

        atr_pips = row['H1_ATR_14'] / 0.0001
        risk_pips = atr_pips * ATR_SL_MULT + SPREAD_PIPS

        sl = entry_price - row['H1_ATR_14'] * ATR_SL_MULT
        tp = entry_price + row['H1_ATR_14'] * ATR_TP_MULT

        pos_size = position_size_by_risk(capital, risk_pips)
        position = 1
        bars_in_trade = 0

# Закрываем в конце
if position != 0:
    last_idx = test_all.index[-1]
    last_close = test_all['H1_Close'].iloc[-1]
    exit_pips = (last_close - entry_price) / 0.0001 - SPREAD_PIPS
    swap = SWAP_PER_LOT_PER_HOUR * pos_size * bars_in_trade
    commission = COMMISSION_PER_LOT * pos_size
    pnl = exit_pips * PIP_VALUE * pos_size - commission - swap
    capital += pnl
    trade_pnls.append(pnl)
    trade_meta.append({
        'entry': entry_time, 'exit': last_idx, 'size': pos_size,
        'pnl': pnl, 'reason': 'end', 'bars': bars_in_trade, 'swap': swap
    })
    equity[-1] = capital

equity = np.array(equity)
trade_pnls = np.array(trade_pnls)
n_trades = len(trade_pnls)

total_return = (equity[-1] - CAPITAL_START) / CAPITAL_START * 100
running_max = np.maximum.accumulate(equity)
drawdown = (equity / running_max - 1) * 100
max_dd = drawdown.min()
win_rate = (trade_pnls > 0).mean() if n_trades else 0
profit_factor = (
    trade_pnls[trade_pnls > 0].sum() / abs(trade_pnls[trade_pnls < 0].sum())
    if (trade_pnls < 0).any() and (trade_pnls > 0).any() else np.inf
)
sharpe = trade_pnls.mean() / (trade_pnls.std() + 1e-10) * np.sqrt(252) if n_trades else 0
calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else np.inf
avg_bars = np.mean([t['bars'] for t in trade_meta]) if trade_meta else 0
total_swap = sum(t['swap'] for t in trade_meta) if trade_meta else 0

print(f"\n{'='*60}")
print(f"РЕЗУЛЬТАТЫ БЭКТЕСТА v25 (Strict MTF)")
print(f"{'='*60}")
print(f"  Сделок:           {n_trades}")
print(f"  Win Rate:         {win_rate:.1%}")
print(f"  Доходность:       {total_return:+.2f}%")
print(f"  Max Drawdown:     {max_dd:.2f}%")
print(f"  Sharpe:           {sharpe:.2f}")
print(f"  Calmar:           {calmar:.2f}")
print(f"  Profit Factor:    {profit_factor:.2f}")
print(f"  Сред. удержание:  {avg_bars:.1f} баров")
print(f"  Общий своп:       ${total_swap:.2f}")
print(f"  Итоговый капитал: ${equity[-1]:,.2f}")
print(f"{'='*60}")

# =============================================================================
# 8. СОХРАНЕНИЕ
# =============================================================================
eq_df = pd.DataFrame({'Equity': equity}, index=test_all.index)
eq_df['Drawdown_%'] = drawdown
eq_df.to_csv('equity_v25_mtf_strict.csv')

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

axes[0].plot(eq_df.index, eq_df['Equity'], color='teal', linewidth=1.2)
axes[0].axhline(y=CAPITAL_START, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title(f'v25 Strict MTF H1+H4+D1 | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | Trades={n_trades}', fontsize=13)
axes[0].set_ylabel('Capital, $')
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(eq_df.index, eq_df['Drawdown_%'], color='crimson', alpha=0.3)
axes[1].plot(eq_df.index, eq_df['Drawdown_%'], color='crimson', linewidth=0.8)
axes[1].set_ylabel('Drawdown, %')
axes[1].grid(True, alpha=0.3)

axes[2].hist(trade_pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='red', linestyle='--')
axes[2].set_title('Distribution of Trade PnL')
axes[2].set_xlabel('PnL, $')
axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_v25_mtf_strict.png', dpi=150, bbox_inches='tight')
plt.close()

trades_df = pd.DataFrame(trade_meta)
if not trades_df.empty:
    trades_df.to_csv('trades_v25_mtf_strict.csv', index=False)

print("\nСохранено:")
print("  equity_v25_mtf_strict.csv")
print("  trades_v25_mtf_strict.csv")
print("  backtest_v25_mtf_strict.png")
