import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ДАННЫЕ + ПРИЗНАКИ (как в v8, честно)
# ==============================================================================
file_path = 'EURUSD_H1_202401020000_202608262100.csv'
df = pd.read_csv(file_path, sep='\t', header=0,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown1', 'Unknown2'])
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].astype(float)

def add_features(data):
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']
    data['SMA_20'] = c.rolling(20).mean(); data['SMA_50'] = c.rolling(50).mean()
    data['EMA_12'] = c.ewm(span=12).mean(); data['EMA_26'] = c.ewm(span=26).mean()
    data['Dist_SMA20'] = (c - data['SMA_20']) / data['SMA_20']
    data['Dist_SMA50'] = (c - data['SMA_50']) / data['SMA_50']
    data['SMA_Spread'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
    pc = c.shift(1); ps = data['SMA_20'].shift(1)
    data['Cross_Up'] = ((c > data['SMA_20']) & (pc <= ps)).astype(int)
    data['Cross_Down'] = ((c < data['SMA_20']) & (pc >= ps)).astype(int)
    d = c.diff(); g = d.where(d > 0, 0).ewm(alpha=1/14).mean(); lo = (-d.where(d < 0, 0)).ewm(alpha=1/14).mean()
    data['RSI_14'] = 100 - 100 / (1 + g / (lo + 1e-10))
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Hist'] = data['MACD'] - data['MACD'].ewm(span=9).mean()
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    data['ATR_Ratio'] = data['ATR_14'] / (data['ATR_14'].rolling(50).mean() + 1e-10)
    bbm = c.rolling(20).mean(); bbs = c.rolling(20).std()
    data['BB_Pos'] = (c - (bbm - 2*bbs)) / (4*bbs + 1e-10)
    data['BB_Width'] = (4*bbs) / (bbm + 1e-10)
    data['Body'] = np.abs(c - o) / (data['ATR_14'] + 1e-10)
    data['UpperWick'] = (h - pd.concat([c,o], axis=1).max(axis=1)) / (data['ATR_14'] + 1e-10)
    data['LowerWick'] = (pd.concat([c,o], axis=1).min(axis=1) - l) / (data['ATR_14'] + 1e-10)
    data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)
    for p in [1,2,3,5,10,20]: data[f'Ret_{p}'] = c.pct_change(p)
    data['MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)
    data['Hour'] = data.index.hour; data['DayOfWeek'] = data.index.dayofweek
    data['London'] = data['Hour'].between(8,15).astype(int)
    data['Overlap'] = data['Hour'].between(13,15).astype(int)
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    a14 = tr.rolling(14).mean()
    pdi = 100*(pdm.rolling(14).mean()/(a14+1e-10)); mdi = 100*(mdm.rolling(14).mean()/(a14+1e-10))
    dx = 100*np.abs(pdi-mdi)/(pdi+mdi+1e-10)
    data['ADX'] = dx.rolling(14).mean(); data['Trend_Strength'] = data['ADX']/50.0
    data['Trend_Direction'] = np.sign(data['SMA_Spread'])
    return data

def add_mtf(data):
    c = data['Close']
    data['H4_SMA_5'] = c.rolling(20).mean(); data['H4_SMA_20'] = c.rolling(80).mean()
    data['H4_Ret'] = c.pct_change(4)
    data['D1_SMA_5'] = c.rolling(120).mean(); data['D1_SMA_20'] = c.rolling(240).mean()
    data['D1_Ret'] = c.pct_change(24)
    data['Dist_H4_SMA5'] = (c-data['H4_SMA_5'])/(data['H4_SMA_5']+1e-10)
    data['Dist_H4_SMA20'] = (c-data['H4_SMA_20'])/(data['H4_SMA_20']+1e-10)
    data['H4_SMA_Cross'] = np.sign(data['H4_SMA_5']-data['H4_SMA_20'])
    data['Dist_D1_SMA5'] = (c-data['D1_SMA_5'])/(data['D1_SMA_5']+1e-10)
    data['Dist_D1_SMA20'] = (c-data['D1_SMA_20'])/(data['D1_SMA_20']+1e-10)
    data['D1_SMA_Cross'] = np.sign(data['D1_SMA_5']-data['D1_SMA_20'])
    return data

df = add_features(df); df = add_mtf(df); df = df.iloc[240:].copy()

feature_cols = ['Dist_SMA20','Dist_SMA50','SMA_Spread','Cross_Up','Cross_Down','RSI_14','MACD_Hist',
    'ATR_Ratio','BB_Pos','BB_Width','Body','UpperWick','LowerWick','Vol_Ratio','MR_Score',
    'Ret_1','Ret_2','Ret_3','Ret_5','Ret_10','Trend_Strength','Trend_Direction',
    'Dist_H4_SMA5','Dist_H4_SMA20','H4_SMA_Cross','H4_Ret','Dist_D1_SMA5','Dist_D1_SMA20',
    'D1_SMA_Cross','D1_Ret','Hour','DayOfWeek','London','Overlap']

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(subset=['Target'], inplace=True)

N_MODELS = 7
ALLOWED = ['range_lo', 'range_hi']
capital_start = 10000
spread_pips = 1.0
pip_value = 10

# ==============================================================================
# 2. WALK-FORWARD: тренируем N_MODELS, сохраняем предсказания КАЖДОЙ модели
# ==============================================================================
# Храним предсказания каждой модели отдельно, выровненные по индексу теста
pred_by_model = {s: pd.Series(dtype=float) for s in range(N_MODELS)}
test_months = pd.date_range('2025-03-01', '2026-07-01', freq='2MS')

for i, test_start in enumerate(test_months):
    test_end = test_start + pd.DateOffset(months=2)
    train = df[df.index < test_start]; test = df[(df.index >= test_start) & (df.index < test_end)]
    if len(train) < 500 or len(test) < 30:
        continue
    X_tr, y_tr = train[feature_cols], train['Target']
    X_te = test[feature_cols]
    for seed in range(N_MODELS):
        m = lgb.LGBMClassifier(
            n_estimators=300 + seed*50, max_depth=3 + (seed % 3),
            learning_rate=0.03, subsample=0.6 + seed*0.05,
            colsample_bytree=0.6 + seed*0.05, min_child_samples=50,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed*7, verbose=-1)
        m.fit(X_tr, y_tr)
        prob = pd.Series(m.predict_proba(X_te)[:, 1], index=test.index)
        pred_by_model[seed] = pd.concat([pred_by_model[seed], prob])
    print(f"  WF {i+1}: done")

# Режимы считаем один раз (детерминированы от цены, не от модели)
regime_series = pd.Series(index=df.index, dtype=object)
test_all = df.loc[[idx for idx in pred_by_model[0].index]]
for idx in pred_by_model[0].index:
    row = df.loc[idx]
    if (row['ATR_Ratio'] > 1.2) and (row['Trend_Strength'] > 0.6): regime_series[idx] = 'trend_hi'
    elif (row['ATR_Ratio'] > 1.2): regime_series[idx] = 'range_hi'
    elif (row['Trend_Strength'] > 0.6): regime_series[idx] = 'trend_lo'
    else: regime_series[idx] = 'range_lo'

next_ret = (df['Close'].shift(-1) - df['Close']) / 0.0001
next_ret = next_ret.loc[pred_by_model[0].index]

# ==============================================================================
# 3. КАЖДАЯ МОДЕЛЬ ТОРГУЕТ САМА ЗА СЕБЯ
# ==============================================================================
# Для каждой модели: свой оптимальный порог, своя эквити
equity_curves = {}
per_model_stats = {}

for seed in range(N_MODELS):
    probs = pred_by_model[seed]
    # оптимальный порог на своих предсказаниях
    best_thr, best_edge = 0.55, -1
    for thr in np.arange(0.50, 0.66, 0.01):
        mask = probs > thr
        if mask.sum() < 10: continue
        wr = (probs[mask].index.map(lambda i: df.loc[i,'Target']) == 1).mean() if mask.sum()>0 else 0
        edge = wr - (1 - wr)
        if edge > best_edge:
            best_edge, best_thr = edge, thr

    capital = capital_start
    eq = [capital]
    pnls = []
    for idx in probs.index:
        if regime_series[idx] not in ALLOWED:
            eq.append(capital); continue
        if probs[idx] > best_thr:
            ret = next_ret[idx] - spread_pips
            conf = (probs[idx] - 0.5) * 2
            pos = min(max(conf*0.3, 0.02), 0.1)
            pnl = ret * pip_value * pos * 10
            capital += pnl; pnls.append(pnl)
        eq.append(capital)
    equity_curves[seed] = np.array(eq)
    p = np.array(pnls)
    wr = (p > 0).mean() if len(p) else 0
    ret = (eq[-1] - capital_start)/capital_start*100
    dd = ((np.array(eq)/np.maximum.accumulate(eq))-1).min()*100
    sharpe = p.mean()/(p.std()+1e-10)*np.sqrt(252) if len(p) else 0
    per_model_stats[seed] = {'thr':best_thr,'n':len(p),'wr':wr,'ret':ret,'dd':dd,'sharpe':sharpe,'final':eq[-1]}

# ==============================================================================
# 4. ОБЩАЯ КОРЗИНА (equal-weight basket): каждая модель = 1/N капитала
# ==============================================================================
# P&L корзины = среднее P&L всех моделей на каждом баре
L = len(equity_curves[0])
basket_pnl_per_step = np.zeros(L - 1)
for seed in range(N_MODELS):
    eq = equity_curves[seed]
    step_pnl = np.diff(eq)  # P&L на каждом шаге
    basket_pnl_per_step += step_pnl / N_MODELS
basket_eq = np.concatenate([[capital_start], capital_start + np.cumsum(basket_pnl_per_step)])

bp = basket_pnl_per_step[basket_pnl_per_step != 0]  # только бары с активными сделками
basket_wr = (bp > 0).mean() if len(bp) else 0
basket_ret = (basket_eq[-1] - capital_start)/capital_start*100
basket_dd = ((basket_eq/np.maximum.accumulate(basket_eq))-1).min()*100
basket_sharpe = bp.mean()/(bp.std()+1e-10)*np.sqrt(252) if len(bp) else 0

# ==============================================================================
# 5. ОТЧЁТ
# ==============================================================================
print(f"\n{'='*68}")
print(f"НЕЗАВИСИМЫЕ ТРЕЙДЕРЫ (каждый сам за себя) + ОБЩАЯ КОРЗИНА")
print(f"{'='*68}")
print(f"{'Модель':<8}{'Thr':>6}{'Trades':>8}{'WR':>8}{'Ret%':>9}{'MaxDD':>9}{'Sharpe':>9}{'Final$':>11}")
print('-'*68)
for seed in range(N_MODELS):
    s = per_model_stats[seed]
    print(f"M{seed+1:<7}{s['thr']:>6.2f}{s['n']:>8}{s['wr']:>7.1%}{s['ret']:>+8.1f}%{s['dd']:>8.1f}%{s['sharpe']:>9.2f}{s['final']:>11,.0f}")
print('-'*68)
print(f"{'КОРЗИНА':<8}{'—':>6}{len(bp):>8}{basket_wr:>7.1%}{basket_ret:>+8.1f}%{basket_dd:>8.1f}%{basket_sharpe:>9.2f}{basket_eq[-1]:>11,.0f}")

# Чемпион
best_seed = max(per_model_stats, key=lambda k: per_model_stats[k]['sharpe'])
print(f"\nЧемпион по Sharpe: M{best_seed+1} (Sharpe={per_model_stats[best_seed]['sharpe']:.2f}, Ret={per_model_stats[best_seed]['ret']:+.1f}%)")
print(f"Корзина vs Чемпион: Ret {basket_ret:+.1f}% vs {per_model_stats[best_seed]['ret']:+.1f}% | "
      f"Sharpe {basket_sharpe:.2f} vs {per_model_stats[best_seed]['sharpe']:.2f} | "
      f"DD {basket_dd:.1f}% vs {per_model_stats[best_seed]['dd']:.1f}%")

# ==============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ==============================================================================
plt.figure(figsize=(14, 8))
idx = list(pred_by_model[0].index)
idx_plot = idx + [idx[-1]]  # для выравнивания длины
for seed in range(N_MODELS):
    plt.plot(idx_plot, equity_curves[seed], linewidth=0.8, alpha=0.5, label=f'M{seed+1}')
plt.plot(idx_plot, basket_eq, color='black', linewidth=2.2, label='КОРЗИНА (basket)')
plt.axhline(y=capital_start, color='gray', linestyle='--', alpha=0.5)
plt.title('Независимые модели + общая корзина (range-режимы)', fontsize=13)
plt.ylabel('Капитал, $'); plt.grid(True, alpha=0.3); plt.legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig('backtest_v9_independent.png', dpi=150, bbox_inches='tight'); plt.close()
print("\nГрафик: backtest_v9_independent.png")

# Сохраняем эквити корзины
basket_df = pd.DataFrame({'Equity': basket_eq}, index=idx_plot)
basket_df['Drawdown_%'] = ((basket_df['Equity']/basket_df['Equity'].cummax())-1)*100
basket_df.to_csv('equity_v9_basket.csv')
print("Эквити корзины: equity_v9_basket.csv")
