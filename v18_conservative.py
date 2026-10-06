import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ДАННЫЕ
# =============================================================================
file_path = 'EURUSD_H1_202401020000_202608262100.csv'
df = pd.read_csv(file_path, sep='\t', header=0,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2'])
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].astype(float)

# =============================================================================
# 2. ПРИЗНАКИ
# =============================================================================
def add_features(data):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']

    for win in [10, 20, 50, 100]:
        data[f'SMA_{win}'] = c.rolling(win).mean()
        data[f'Dist_SMA{win}'] = (c - data[f'SMA_{win}']) / data[f'SMA_{win}']
    data['EMA_12'] = c.ewm(span=12).mean()
    data['EMA_26'] = c.ewm(span=26).mean()
    data['SMA_Spread_20_50'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
    data['SMA_Spread_10_20'] = (data['SMA_10'] - data['SMA_20']) / data['SMA_20']

    prev_c = c.shift(1)
    prev_sma = data['SMA_20'].shift(1)
    data['Cross_Up'] = ((c > data['SMA_20']) & (prev_c <= prev_sma)).astype(int)
    data['Cross_Down'] = ((c < data['SMA_20']) & (prev_c >= prev_sma)).astype(int)

    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    data['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    data['RSI_7'] = 100 - 100 / (1 + delta.where(delta > 0, 0).ewm(alpha=1/7).mean() /
                                   (delta.where(delta < 0, 0).abs().ewm(alpha=1/7).mean() + 1e-10))
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    for win in [14, 50]:
        data[f'ATR_{win}'] = tr.rolling(win).mean()
    data['ATR_Ratio'] = data['ATR_14'] / (data['ATR_50'] + 1e-10)
    data['ATR_Pct'] = data['ATR_14'] / c

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data['BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    data['BB_Width'] = (4*bb_std) / (bb_mid + 1e-10)
    data['BB_Touch_Upper'] = (c >= bb_mid + 2*bb_std).astype(int)
    data['BB_Touch_Lower'] = (c <= bb_mid - 2*bb_std).astype(int)

    data['Body'] = np.abs(c - o)
    data['Body_Pct'] = data['Body'] / (data['ATR_14'] + 1e-10)
    data['UpperWick'] = (h - pd.concat([c,o], axis=1).max(axis=1)) / (data['ATR_14'] + 1e-10)
    data['LowerWick'] = (pd.concat([c,o], axis=1).min(axis=1) - l) / (data['ATR_14'] + 1e-10)
    data['Range'] = (h - l) / (data['ATR_14'] + 1e-10)

    data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)
    data['Vol_Ratio_50'] = data['Volume'] / (data['Volume'].rolling(50).mean() + 1)

    for p in [1, 2, 3, 5, 10, 20, 50]:
        data[f'Ret_{p}'] = c.pct_change(p)
    for p in [1, 2, 3, 5]:
        data[f'Lag_Ret_{p}'] = c.pct_change(p).shift(p)

    data['MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)
    data['MR_Score_50'] = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-10)

    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['London'] = data['Hour'].between(8, 15).astype(int)
    data['Overlap'] = data['Hour'].between(13, 15).astype(int)
    data['Asia'] = data['Hour'].between(0, 7).astype(int)
    data['US'] = data['Hour'].between(16, 21).astype(int)
    data['Monday'] = (data['DayOfWeek'] == 0).astype(int)
    data['Friday'] = (data['DayOfWeek'] == 4).astype(int)
    data['Avoid_Time'] = ((data['DayOfWeek'] == 4) & (data['Hour'] >= 20)) | \
                         ((data['DayOfWeek'] == 0) & (data['Hour'] <= 4)).astype(int)

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    data['ADX'] = dx.rolling(14).mean()
    data['DI_Diff'] = plus_di - minus_di
    data['Trend_Strength'] = data['ADX'] / 50.0
    data['Trend_Direction'] = np.sign(data['SMA_Spread_20_50'])

    return data


def add_mtf(data):
    c = data['Close']
    data['H4_SMA_5'] = c.rolling(20).mean()
    data['H4_SMA_20'] = c.rolling(80).mean()
    data['H4_Ret'] = c.pct_change(4)
    data['D1_SMA_5'] = c.rolling(120).mean()
    data['D1_SMA_20'] = c.rolling(240).mean()
    data['D1_Ret'] = c.pct_change(24)
    data['W1_SMA_5'] = c.rolling(600).mean()
    data['W1_Ret'] = c.pct_change(120)

    data['Dist_H4_SMA5'] = (c - data['H4_SMA_5']) / (data['H4_SMA_5'] + 1e-10)
    data['Dist_H4_SMA20'] = (c - data['H4_SMA_20']) / (data['H4_SMA_20'] + 1e-10)
    data['H4_SMA_Cross'] = np.sign(data['H4_SMA_5'] - data['H4_SMA_20'])
    data['Dist_D1_SMA5'] = (c - data['D1_SMA_5']) / (data['D1_SMA_5'] + 1e-10)
    data['Dist_D1_SMA20'] = (c - data['D1_SMA_20']) / (data['D1_SMA_20'] + 1e-10)
    data['D1_SMA_Cross'] = np.sign(data['D1_SMA_5'] - data['D1_SMA_20'])
    data['Dist_W1_SMA5'] = (c - data['W1_SMA_5']) / (data['W1_SMA_5'] + 1e-10)
    return data


df = add_features(df)
df = add_mtf(df)
df = df.iloc[600:].copy()

feature_cols = [c for c in df.columns if c not in [
    'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Unknown1', 'Unknown2', 'Target', 'Regime'
]]

# =============================================================================
# 3. РЕЖИМЫ
# =============================================================================
def regime_of(r):
    if (r['ATR_Ratio'] > 1.2) and (r['Trend_Strength'] > 0.6):
        return 'trend_hi'
    elif r['ATR_Ratio'] > 1.2:
        return 'range_hi'
    elif r['Trend_Strength'] > 0.6:
        return 'trend_lo'
    else:
        return 'range_lo'

df['Regime'] = df.apply(regime_of, axis=1)
regimes = ['trend_hi', 'range_hi', 'trend_lo', 'range_lo']

# =============================================================================
# 4. TARGET: mean-reversion flip
# =============================================================================
# Цель: предсказать, отскочит ли цена в сторону средней
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# Смещаем target к MR: если цена сильно ниже средней — скорее вырастет
mr_bias = np.where(
    df['MR_Score'] > 1.5, 0,
    np.where(df['MR_Score'] < -1.5, 1, df['Target'])
)
df['Target'] = mr_bias
df.dropna(subset=['Target'], inplace=True)

print(f"Данные: {len(df)} строк, признаков: {len(feature_cols)}")
print(f"Режимы:\n{df['Regime'].value_counts()}")

# =============================================================================
# 5. WALK-FORWARD ENSEMBLE
# =============================================================================
N_MODELS = 7

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

    probs_list = []
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
        prob = m.predict_proba(X_te)[:, 1]
        probs_list.append(prob)

        # Edge по режимам
        prob_s = pd.Series(prob, index=test.index)
        for r in regimes:
            sub = test[test['Regime'] == r]
            if len(sub) > 0:
                preds = (prob_s[sub.index] > 0.5).astype(int)
                correct = (preds.values == sub['Target'].values).sum()
                edge[seed][r].append(correct / len(sub))

    avg_prob = np.mean(probs_list, axis=0)
    pred_by_model['ensemble'] = pd.concat([
        pred_by_model.get('ensemble', pd.Series(dtype=float)),
        pd.Series(avg_prob, index=test.index)
    ])

    for s in range(N_MODELS):
        pred_by_model[s] = pd.concat([
            pred_by_model[s],
            pd.Series(probs_list[s], index=test.index)
        ])

    print(f"  WF {i+1}: {test_start.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}")

# Усредняем edge
for m in range(N_MODELS):
    for r in regimes:
        edge[m][r] = np.mean(edge[m][r]) if edge[m][r] else 0.5

# =============================================================================
# 6. ФИНАЛЬНЫЙ КОНСЕНСУС С ДИНАМИЧЕСКИМ ВЗВЕШИВАНИЕМ
# =============================================================================
all_idx = sorted(pred_by_model['ensemble'].index)
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

# Следующий барный return
test_all['Next_Return'] = (test_all['Close'].shift(-1) - test_all['Close']) / 0.0001
test_all.dropna(subset=['Next_Return'], inplace=True)

print(f"\nТестовых баров: {len(test_all)}")

# =============================================================================
# 7. БЭКТЕСТ: ТОЛЬКО LONG, СТОП/ТЕЙК, КОМИССИЯ, ДИНАМИЧЕСКИЙ РАЗМЕР
# =============================================================================
# Подбор порога по режиму
regime_thr = {}
for r in test_all['Regime'].unique():
    rd = test_all[test_all['Regime'] == r]
    if len(rd) < 30:
        regime_thr[r] = 0.55
        continue
    best_thr, best_ev = 0.55, -999
    for thr in np.arange(0.55, 0.75, 0.01):
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
MAX_RISK_PCT = 0.015
MIN_HOLD_BARS = 3


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
    # Mark-to-market
    if position != 0:
        unrealized_pips = (row['Close'] - entry_price) / 0.0001 * position
        unrealized = unrealized_pips * PIP_VALUE * pos_size
        current_equity = capital + unrealized
    else:
        current_equity = capital

    equity.append(current_equity)

    # Проверяем закрытие позиции
    if position != 0:
        bars_in_trade += 1
        high = row['High']
        low = row['Low']
        exit_reason = None

        if position == 1:
            if low <= sl:
                exit_pips = (sl - entry_price) / 0.0001 - SPREAD_PIPS
                exit_reason = 'sl'
            elif high >= tp:
                exit_pips = (tp - entry_price) / 0.0001 - SPREAD_PIPS
                exit_reason = 'tp'
            elif bars_in_trade >= MIN_HOLD_BARS and row['Prob'] < 0.45:
                # Выход по развороту сигнала
                exit_pips = (row['Close'] - entry_price) / 0.0001 - SPREAD_PIPS
                exit_reason = 'signal'
            else:
                continue
        else:
            continue  # short запрещены

        commission = COMMISSION_PER_LOT * pos_size
        pnl = exit_pips * PIP_VALUE * pos_size - commission
        capital += pnl
        trade_pnls.append(pnl)
        trade_meta.append({
            'entry': entry_time, 'exit': idx, 'dir': position,
            'size': pos_size, 'pnl': pnl, 'reason': exit_reason,
            'bars': bars_in_trade
        })
        position = 0
        entry_price = 0
        entry_time = None
        pos_size = 0
        sl = 0
        tp = 0
        bars_in_trade = 0

    # Открываем новую позицию (только long)
    if position == 0:
        # Фильтры
        if row['Avoid_Time']:
            continue
        if row['Regime'] not in ['range_lo', 'range_hi', 'trend_lo']:
            continue

        thr = regime_thr.get(row['Regime'], 0.55)
        if row['Prob'] <= thr:
            continue

        # Дополнительный фильтр: не входим, если цена уже у верхней BB
        if row['BB_Pos'] > 0.85:
            continue

        direction = 1
        entry_price = row['Close']
        entry_time = idx

        atr_pips = row['ATR_14'] / 0.0001
        risk_pips = atr_pips * ATR_SL_MULT + SPREAD_PIPS

        sl = entry_price - row['ATR_14'] * ATR_SL_MULT
        tp = entry_price + row['ATR_14'] * ATR_TP_MULT

        pos_size = position_size_by_risk(capital, risk_pips)
        position = direction
        bars_in_trade = 0

# Закрываем в конце
if position != 0:
    last_idx = test_all.index[-1]
    last_close = test_all['Close'].iloc[-1]
    exit_pips = (last_close - entry_price) / 0.0001 - SPREAD_PIPS
    commission = COMMISSION_PER_LOT * pos_size
    pnl = exit_pips * PIP_VALUE * pos_size - commission
    capital += pnl
    trade_pnls.append(pnl)
    trade_meta.append({
        'entry': entry_time, 'exit': last_idx, 'dir': position,
        'size': pos_size, 'pnl': pnl, 'reason': 'end', 'bars': bars_in_trade
    })
    equity[-1] = capital

equity = np.array(equity)
trade_pnls = np.array(trade_pnls)
n_trades = len(trade_pnls)

# Метрики
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

print(f"\n{'='*60}")
print(f"РЕЗУЛЬТАТЫ БЭКТЕСТА v18")
print(f"{'='*60}")
print(f"  Сделок:           {n_trades}")
print(f"  Win Rate:         {win_rate:.1%}")
print(f"  Доходность:       {total_return:+.2f}%")
print(f"  Max Drawdown:     {max_dd:.2f}%")
print(f"  Sharpe:           {sharpe:.2f}")
print(f"  Calmar:           {calmar:.2f}")
print(f"  Profit Factor:    {profit_factor:.2f}")
print(f"  Сред. удержание:  {avg_bars:.1f} баров")
print(f"  Итоговый капитал: ${equity[-1]:,.2f}")
print(f"{'='*60}")

# =============================================================================
# 8. СОХРАНЕНИЕ
# =============================================================================
eq_df = pd.DataFrame({'Equity': equity}, index=test_all.index)
eq_df['Drawdown_%'] = drawdown
eq_df.to_csv('equity_v18_conservative.csv')

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

axes[0].plot(eq_df.index, eq_df['Equity'], color='teal', linewidth=1.2)
axes[0].axhline(y=CAPITAL_START, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title(f'v18 Conservative Long-Only | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | Trades={n_trades}', fontsize=13)
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
plt.savefig('backtest_v18_conservative.png', dpi=150, bbox_inches='tight')
plt.close()

trades_df = pd.DataFrame(trade_meta)
if not trades_df.empty:
    trades_df.to_csv('trades_v18_conservative.csv', index=False)

print("\nСохранено:")
print("  equity_v18_conservative.csv")
print("  trades_v18_conservative.csv")
print("  backtest_v18_conservative.png")
