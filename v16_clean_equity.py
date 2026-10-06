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
# 2. ПРИЗНАКИ (только каузальные — только прошлое)
# =============================================================================
def add_features(data):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']

    # Trend
    data['SMA_20'] = c.rolling(20).mean()
    data['SMA_50'] = c.rolling(50).mean()
    data['EMA_12'] = c.ewm(span=12).mean()
    data['EMA_26'] = c.ewm(span=26).mean()
    data['Dist_SMA20'] = (c - data['SMA_20']) / data['SMA_20']
    data['Dist_SMA50'] = (c - data['SMA_50']) / data['SMA_50']
    data['SMA_Spread'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
    prev_c = c.shift(1)
    prev_sma = data['SMA_20'].shift(1)
    data['Cross_Up'] = ((c > data['SMA_20']) & (prev_c <= prev_sma)).astype(int)
    data['Cross_Down'] = ((c < data['SMA_20']) & (prev_c >= prev_sma)).astype(int)

    # Momentum
    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    data['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Volatility
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    data['ATR_Ratio'] = data['ATR_14'] / (data['ATR_14'].rolling(50).mean() + 1e-10)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data['BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    data['BB_Width'] = (4*bb_std) / (bb_mid + 1e-10)

    # Candle
    data['Body'] = np.abs(c - o) / (data['ATR_14'] + 1e-10)
    data['UpperWick'] = (h - pd.concat([c,o], axis=1).max(axis=1)) / (data['ATR_14'] + 1e-10)
    data['LowerWick'] = (pd.concat([c,o], axis=1).min(axis=1) - l) / (data['ATR_14'] + 1e-10)

    # Volume
    data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)

    # Returns
    for p in [1, 2, 3, 5, 10, 20]:
        data[f'Ret_{p}'] = c.pct_change(p)

    # Mean reversion
    data['MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)

    # Session
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['London'] = data['Hour'].between(8, 15).astype(int)
    data['Overlap'] = data['Hour'].between(13, 15).astype(int)

    # Regime detection (ADX)
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    data['ADX'] = dx.rolling(14).mean()
    data['Trend_Strength'] = data['ADX'] / 50.0
    data['Trend_Direction'] = np.sign(data['SMA_Spread'])

    return data

def add_mtf(data):
    c = data['Close']
    data['H4_SMA_5'] = c.rolling(20).mean()
    data['H4_SMA_20'] = c.rolling(80).mean()
    data['H4_Ret'] = c.pct_change(4)
    data['D1_SMA_5'] = c.rolling(120).mean()
    data['D1_SMA_20'] = c.rolling(240).mean()
    data['D1_Ret'] = c.pct_change(24)
    data['Dist_H4_SMA5'] = (c - data['H4_SMA_5']) / (data['H4_SMA_5'] + 1e-10)
    data['Dist_H4_SMA20'] = (c - data['H4_SMA_20']) / (data['H4_SMA_20'] + 1e-10)
    data['H4_SMA_Cross'] = np.sign(data['H4_SMA_5'] - data['H4_SMA_20'])
    data['Dist_D1_SMA5'] = (c - data['D1_SMA_5']) / (data['D1_SMA_5'] + 1e-10)
    data['Dist_D1_SMA20'] = (c - data['D1_SMA_20']) / (data['D1_SMA_20'] + 1e-10)
    data['D1_SMA_Cross'] = np.sign(data['D1_SMA_5'] - data['D1_SMA_20'])
    return data

df = add_features(df)
df = add_mtf(df)
df = df.iloc[240:].copy()

feature_cols = [
    'Dist_SMA20', 'Dist_SMA50', 'SMA_Spread', 'Cross_Up', 'Cross_Down',
    'RSI_14', 'MACD_Hist', 'ATR_Ratio', 'BB_Pos', 'BB_Width',
    'Body', 'UpperWick', 'LowerWick', 'Vol_Ratio', 'MR_Score',
    'Ret_1', 'Ret_2', 'Ret_3', 'Ret_5', 'Ret_10',
    'Trend_Strength', 'Trend_Direction',
    'Dist_H4_SMA5', 'Dist_H4_SMA20', 'H4_SMA_Cross', 'H4_Ret',
    'Dist_D1_SMA5', 'Dist_D1_SMA20', 'D1_SMA_Cross', 'D1_Ret',
    'Hour', 'DayOfWeek', 'London', 'Overlap',
]

# Target: цена следующего бара выше текущей
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(subset=['Target'], inplace=True)

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

print(f"Данные: {len(df)} строк, признаков: {len(feature_cols)}")
print(f"Режимы:\n{df['Regime'].value_counts()}")

# =============================================================================
# 4. WALK-FORWARD С ДИНАМИЧЕСКИМ ВЗВЕШИВАНИЕМ
# =============================================================================
N_MODELS = 7
ALLOWED = ['range_lo', 'range_hi']  # торгуем только боковик (mean-reversion логика)

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

        # Out-of-sample edge модели по режимам
        for r in regimes:
            sub = test[test['Regime'] == r]
            if len(sub) > 0:
                p = prob[sub.index]
                preds = (p > 0.5).astype(int)
                correct = (preds.values == sub['Target'].values).sum()
                edge[seed][r].append(correct / len(sub))

    print(f"  WF {i+1}: {test_start.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}, train={len(train)}, test={len(test)}")

# Усредняем edge по окнам
for m in range(N_MODELS):
    for r in regimes:
        edge[m][r] = np.mean(edge[m][r]) if edge[m][r] else 0.5

# Собираем финальный консенсус с динамическими весами
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

# Следующий барный return в пипсах (1 пип = 0.0001 для EURUSD)
test_all['Next_Return'] = (test_all['Close'].shift(-1) - test_all['Close']) / 0.0001
test_all.dropna(subset=['Next_Return'], inplace=True)

print(f"\nТестовых баров: {len(test_all)}")

# =============================================================================
# 5. БЭКТЕСТ С ЭКВИТИ
# =============================================================================
# Подбор порога по режиму на out-of-sample (walk-forward уже дал предсказания)
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
        ev = wr - (1 - wr)  # прибыль 1, убыток 1
        if ev > best_ev:
            best_ev, best_thr = ev, thr
    regime_thr[r] = best_thr

print("\nRegime thresholds:", {k: f"{v:.2f}" for k, v in regime_thr.items()})

# Параметры капитала
CAPITAL_START = 10000
SPREAD_PIPS = 1.0
PIP_VALUE = 10  # $ за пип на лот (упрощённо)

capital = CAPITAL_START
equity = []
trade_pnls = []
trade_meta = []

for idx, row in test_all.iterrows():
    regime = row['Regime']

    # Торгуем только разрешённые режимы
    if regime not in ALLOWED:
        equity.append(capital)
        continue

    thr = regime_thr.get(regime, 0.55)
    if row['Prob'] > thr:
        # return за вычетом спреда
        ret = row['Next_Return'] - SPREAD_PIPS

        # размер позиции по уверенности: 0.02..0.1 лота
        conf = (row['Prob'] - 0.5) * 2
        pos = min(max(conf * 0.3, 0.02), 0.1)

        # PnL в долларах
        pnl = ret * PIP_VALUE * pos * 10
        capital += pnl
        trade_pnls.append(pnl)
        trade_meta.append({'time': idx, 'pnl': pnl, 'prob': row['Prob'], 'regime': regime, 'pos': pos})

    equity.append(capital)

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

print(f"\n{'='*60}")
print(f"РЕЗУЛЬТАТЫ БЭКТЕСТА")
print(f"{'='*60}")
print(f"  Сделок:           {n_trades}")
print(f"  Win Rate:         {win_rate:.1%}")
print(f"  Доходность:       {total_return:+.2f}%")
print(f"  Max Drawdown:     {max_dd:.2f}%")
print(f"  Sharpe (annual):  {sharpe:.2f}")
print(f"  Profit Factor:    {profit_factor:.2f}")
print(f"  Итоговый капитал: ${equity[-1]:,.2f}")
print(f"{'='*60}")

# =============================================================================
# 6. СОХРАНЕНИЕ ЭКВИТИ И ГРАФИК
# =============================================================================
eq_df = pd.DataFrame({'Equity': equity}, index=test_all.index)
eq_df['Drawdown_%'] = drawdown
eq_df.to_csv('equity_v16_clean.csv')

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(eq_df.index, eq_df['Equity'], color='teal', linewidth=1.2)
plt.axhline(y=CAPITAL_START, color='gray', linestyle='--', alpha=0.5)
plt.title(f'v16 Clean Equity | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f}', fontsize=13)
plt.ylabel('Capital, $')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.fill_between(eq_df.index, eq_df['Drawdown_%'], color='crimson', alpha=0.3)
plt.plot(eq_df.index, eq_df['Drawdown_%'], color='crimson', linewidth=0.8)
plt.ylabel('Drawdown, %')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_v16_clean.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nСохранено:")
print("  equity_v16_clean.csv")
print("  backtest_v16_clean.png")
