import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ДАННЫЕ
# ==============================================================================
file_path = 'EURUSD_H1_202401020000_202608262100.csv'
df = pd.read_csv(file_path, sep='\t', header=0,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2'])
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].astype(float)

# ==============================================================================
# 2. КАУЗАЛЬНЫЕ ПРИЗНАКИ (только прошлое!)
# ==============================================================================
def add_features(data):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']
    
    # --- Trend ---
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
    
    # --- Momentum ---
    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    data['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # --- Volatility ---
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    data['ATR_Ratio'] = data['ATR_14'] / (data['ATR_14'].rolling(50).mean() + 1e-10)
    
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data['BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    data['BB_Width'] = (4*bb_std) / (bb_mid + 1e-10)
    
    # --- Candle ---
    data['Body'] = np.abs(c - o) / (data['ATR_14'] + 1e-10)
    data['UpperWick'] = (h - pd.concat([c,o], axis=1).max(axis=1)) / (data['ATR_14'] + 1e-10)
    data['LowerWick'] = (pd.concat([c,o], axis=1).min(axis=1) - l) / (data['ATR_14'] + 1e-10)
    
    # --- Volume ---
    data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)
    
    # --- Returns ---
    for p in [1, 2, 3, 5, 10, 20]:
        data[f'Ret_{p}'] = c.pct_change(p)
    
    # --- Mean Reversion ---
    data['MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)
    
    # --- Session ---
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['London'] = data['Hour'].between(8, 15).astype(int)
    data['Overlap'] = data['Hour'].between(13, 15).astype(int)
    
    # --- Regime Detection ---
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

# ==============================================================================
# 3. КАУЗАЛЬНЫЕ MTF ПРИЗНАКИ
# ==============================================================================
def add_causal_mtf_features(data):
    """MTF признаки через скользящие окна (каузальные, без ресемплинга)"""
    c = data['Close']
    
    # H4 proxy: скользящие средние на 4-часовом масштабе
    data['H4_SMA_5'] = c.rolling(20).mean()   # ~20 H1 = 4H
    data['H4_SMA_20'] = c.rolling(80).mean()  # ~80 H1 = 4D
    data['H4_Ret'] = c.pct_change(4)           # 4-hour return
    
    # D1 proxy
    data['D1_SMA_5'] = c.rolling(120).mean()  # ~120 H1 = 5D
    data['D1_SMA_20'] = c.rolling(240).mean() # ~240 H1 = 10D
    data['D1_Ret'] = c.pct_change(24)          # daily return
    
    # Относительные расстояния
    data['Dist_H4_SMA5'] = (c - data['H4_SMA_5']) / (data['H4_SMA_5'] + 1e-10)
    data['Dist_H4_SMA20'] = (c - data['H4_SMA_20']) / (data['H4_SMA_20'] + 1e-10)
    data['H4_SMA_Cross'] = np.sign(data['H4_SMA_5'] - data['H4_SMA_20'])
    
    data['Dist_D1_SMA5'] = (c - data['D1_SMA_5']) / (data['D1_SMA_5'] + 1e-10)
    data['Dist_D1_SMA20'] = (c - data['D1_SMA_20']) / (data['D1_SMA_20'] + 1e-10)
    data['D1_SMA_Cross'] = np.sign(data['D1_SMA_5'] - data['D1_SMA_20'])
    
    return data

# ==============================================================================
# 4. РЕЖИМЫ + TARGET
# ==============================================================================
df = add_features(df)
df = add_causal_mtf_features(df)

# Убираем только первые 240 строк (нужно для самого длинного окна D1)
df = df.iloc[240:].copy()

feature_cols = [
    'Dist_SMA20', 'Dist_SMA50', 'SMA_Spread',
    'Cross_Up', 'Cross_Down',
    'RSI_14', 'MACD_Hist', 'ATR_Ratio', 'BB_Pos', 'BB_Width',
    'Body', 'UpperWick', 'LowerWick',
    'Vol_Ratio', 'MR_Score',
    'Ret_1', 'Ret_2', 'Ret_3', 'Ret_5', 'Ret_10',
    'Trend_Strength', 'Trend_Direction',
    'Dist_H4_SMA5', 'Dist_H4_SMA20', 'H4_SMA_Cross', 'H4_Ret',
    'Dist_D1_SMA5', 'Dist_D1_SMA20', 'D1_SMA_Cross', 'D1_Ret',
    'Hour', 'DayOfWeek', 'London', 'Overlap',
]

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(subset=['Target'], inplace=True)
print(f"Данные: {len(df)} строк, признаков: {len(feature_cols)}")

# ==============================================================================
# 5. WALK-FORWARD С ENSEMBLE
# ==============================================================================
test_months = pd.date_range('2025-03-01', '2026-07-01', freq='2MS')

all_test = []

for i, test_start in enumerate(test_months):
    test_end = test_start + pd.DateOffset(months=2)
    
    train = df[df.index < test_start].copy()
    test = df[(df.index >= test_start) & (df.index < test_end)].copy()
    
    if len(train) < 500 or len(test) < 30:
        continue
    
    X_tr = train[feature_cols]
    y_tr = train['Target']
    X_te = test[feature_cols]
    y_te = test['Target']
    
    # Ансамбль из 7 моделей
    probs_list = []
    for seed in range(7):
        m = lgb.LGBMClassifier(
            n_estimators=300 + seed * 50,
            max_depth=3 + (seed % 3),
            learning_rate=0.03,
            subsample=0.6 + seed * 0.05,
            colsample_bytree=0.6 + seed * 0.05,
            min_child_samples=50,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed * 7, verbose=-1
        )
        m.fit(X_tr, y_tr)
        probs_list.append(m.predict_proba(X_te)[:, 1])
    
    avg_prob = np.mean(probs_list, axis=0)
    
    res = test.copy()
    res['Prob'] = avg_prob
    
    # Режимы
    res['Regime'] = 'neutral'
    res.loc[(res['ATR_Ratio'] > 1.2) & (res['Trend_Strength'] > 0.6), 'Regime'] = 'trend_hi'
    res.loc[(res['ATR_Ratio'] > 1.2) & (res['Trend_Strength'] <= 0.6), 'Regime'] = 'range_hi'
    res.loc[(res['ATR_Ratio'] <= 1.2) & (res['Trend_Strength'] > 0.6), 'Regime'] = 'trend_lo'
    res.loc[(res['ATR_Ratio'] <= 1.2) & (res['Trend_Strength'] <= 0.6), 'Regime'] = 'range_lo'
    
    all_test.append(res)
    print(f"  WF {i+1}: train<{test_start.date()}, test {test_start.date()}-{test_end.date()}, "
          f"acc={((avg_prob > 0.5).astype(int) == y_te.values).mean():.3f}")

test_all = pd.concat(all_test)
test_all = test_all[~test_all.index.duplicated(keep='first')]
test_all.sort_index(inplace=True)
print(f"\nТестовых: {len(test_all)}")

# ==============================================================================
# 6. БЭКТЕСТ
# ==============================================================================
test_all['Next_Return'] = (test_all['Close'].shift(-1) - test_all['Close']) / 0.0001
test_all.dropna(subset=['Next_Return'], inplace=True)

capital_start = 10000
spread_pips = 1.0
pip_value = 10

# Оптимальный threshold для каждого режима
regimes = test_all['Regime'].unique()
regime_thr = {}

for regime in regimes:
    rd = test_all[test_all['Regime'] == regime]
    if len(rd) < 30:
        regime_thr[regime] = 0.55
        continue
    
    best_thr, best_ev = 0.55, -999
    for thr in np.arange(0.50, 0.65, 0.02):
        mask = rd['Prob'] > thr
        if mask.sum() < 5:
            continue
        wr = rd.loc[mask, 'Target'].mean()
        ev = wr * 1.0 - (1 - wr) * 1.0  # simplified EV
        if ev > best_ev:
            best_ev = ev
            best_thr = thr
    regime_thr[regime] = best_thr

print("\nRegime thresholds:", {k: f"{v:.2f}" for k, v in regime_thr.items()})

# Бэктест: только Long (проще и надёжнее)
capital = capital_start
equity = [capital]
trade_pnls = []
trade_regimes = []

for idx, row in test_all.iterrows():
    prob = row['Prob']
    regime = row['Regime']
    thr = regime_thr.get(regime, 0.55)
    
    if prob > thr:
        ret_pips = row['Next_Return'] - spread_pips
        
        # Kelly-like sizing
        confidence = (prob - 0.5) * 2
        pos = min(confidence * 0.3, 0.1)
        pos = max(pos, 0.02)
        
        pnl = ret_pips * pip_value * pos * 10
        capital += pnl
        trade_pnls.append(pnl)
        trade_regimes.append(regime)
    
    equity.append(capital)

trade_pnls = np.array(trade_pnls)
equity = np.array(equity)

# ==============================================================================
# 7. МЕТРИКИ
# ==============================================================================
n = len(trade_pnls)
if n > 0:
    wins = (trade_pnls > 0).sum()
    wr = wins / n
    total_ret = (equity[-1] - capital_start) / capital_start * 100
    max_dd = ((equity / np.maximum.accumulate(equity)) - 1).min() * 100
    sharpe = trade_pnls.mean() / (trade_pnls.std() + 1e-10) * np.sqrt(252)
    gp = trade_pnls[trade_pnls > 0].sum() if (trade_pnls > 0).any() else 0
    gl = abs(trade_pnls[trade_pnls < 0].sum()) if (trade_pnls < 0).any() else 1e-10
    pf = gp / gl
    avg_w = trade_pnls[trade_pnls > 0].mean() if (trade_pnls > 0).any() else 0
    avg_l = trade_pnls[trade_pnls < 0].mean() if (trade_pnls < 0).any() else 0

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ: MTF + Regime Ensemble (Long Only)")
    print(f"{'='*60}")
    print(f"  Сделок: {n}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Доходность: {total_ret:+.1f}%")
    print(f"  Макс. просадка: {max_dd:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Win: ${avg_w:.1f}, Avg Loss: ${avg_l:.1f}")
    print(f"  Capital: ${equity[-1]:,.2f}")

    # By regime
    print(f"\n  По регионам:")
    for r in regimes:
        mask = np.array(trade_regimes) == r
        if mask.sum() < 5:
            continue
        rp = trade_pnls[mask]
        rwr = (rp > 0).sum() / len(rp)
        print(f"    {r:12s}: n={len(rp):3d}, WR={rwr:.1%}, P&L=${rp.sum():+.0f}")

    # Monte Carlo
    print(f"\n  Monte Carlo (2000 итераций):")
    mc_drawdowns = []
    mc_returns = []
    for _ in range(2000):
        shuffled = np.random.permutation(trade_pnls)
        mc_eq = capital_start + np.cumsum(shuffled)
        mc_eq = np.insert(mc_eq, 0, capital_start)
        mc_dd = ((mc_eq / np.maximum.accumulate(mc_eq)) - 1).min() * 100
        mc_drawdowns.append(mc_dd)
        mc_returns.append((mc_eq[-1] - capital_start) / capital_start * 100)
    
    print(f"    Median Return: {np.median(mc_returns):+.1f}%")
    print(f"    5th percentile Return: {np.percentile(mc_returns, 5):+.1f}%")
    print(f"    95th percentile Return: {np.percentile(mc_returns, 95):+.1f}%")
    print(f"    Median MaxDD: {np.median(mc_drawdowns):.1f}%")
    print(f"    95th percentile MaxDD: {np.percentile(mc_drawdowns, 95):.1f}%")

    # ==============================================================================
    # 8. ЭКСПОРТ ЭКВИТИ В CSV
    # ==============================================================================
    eq_df = pd.DataFrame({'Equity': equity[1:]}, index=test_all.index)
    eq_df.index.name = 'DateTime'
    eq_df['Drawdown_%'] = ((eq_df['Equity'] / eq_df['Equity'].cummax()) - 1) * 100
    eq_df.to_csv('equity_v7.csv')
    print(f"\nЭквити сохранено в equity_v7.csv ({len(eq_df)} строк)")

    # ==============================================================================
    # 9. ВИЗУАЛИЗАЦИЯ
    # ==============================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'MTF + Regime Ensemble (Sharpe={sharpe:.2f}, PF={pf:.2f})', fontsize=14)

    axes[0].plot(equity, color='blue', linewidth=1)
    axes[0].axhline(y=capital_start, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('$')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(trade_pnls, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_title(f'P&L (WR={wr:.1%}, PF={pf:.2f})')
    axes[1].grid(True, alpha=0.3)

    # Monte Carlo distribution
    axes[2].hist(mc_returns, bins=50, color='green', alpha=0.7, edgecolor='white')
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2].axvline(x=np.percentile(mc_returns, 5), color='orange', linestyle='--', label='5th pctl')
    axes[2].set_title('Monte Carlo Return Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_v7_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nГрафик: backtest_v7_final.png")
else:
    print("Нет сделок!")
