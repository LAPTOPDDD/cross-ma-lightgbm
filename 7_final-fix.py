import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import math

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. Synthetic data generator (run if real CSV not found)
# ----------------------------------------------------------------------
def generate_synthetic_h1(n_bars=10000):
    np.random.seed(42)
    t = pd.date_range('2024-01-02', periods=n_bars, freq='h')
    base_price = 1.1000
    rets = np.random.normal(0, 0.0008, n_bars)
    prices = base_price * np.cumprod(1 + rets)
    highs = prices * (1 + np.abs(np.random.normal(0, 0.0003, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.0003, n_bars)))
    opens = prices * (1 + np.random.normal(0, 0.0004, n_bars))
    vols = np.random.randint(500, 5000, n_bars)
    data = pd.DataFrame({
        'Date': t.strftime('%Y.%m.%d'),
        'Time': t.strftime('%H:%M:%S'),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'TickVol': vols,
        'Vol': vols * 10,
        'Spread': np.random.randint(1, 3, n_bars)
    })
    return data

# ----------------------------------------------------------------------
# 2. Load CSV or generate synthetic
# ----------------------------------------------------------------------
CSV_PATH = 'EURUSD_H1_202401020000_202608262100.csv'
if os.path.exists(CSV_PATH):
    raw = pd.read_csv(CSV_PATH, sep='\t', skiprows=1,
                      names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
else:
    print('CSV not found – using synthetic data for demonstration.')
    raw = generate_synthetic_h1()

raw['DateTime'] = pd.to_datetime(raw['Date'].astype(str) + ' ' + raw['Time'].astype(str))
raw.set_index('DateTime', inplace=True)
raw.sort_index(inplace=True)

for c in ['Open','High','Low','Close','TickVol','Vol','Spread']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')

print(f'Loaded {len(raw)} bars from {raw.index[0].date()} to {raw.index[-1].date()}')
df = raw.copy()

# ----------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------
def add_features(d):
    c = d['Close']; h = d['High']; l = d['Low']; o = d['Open']

    d['SMA20'] = c.rolling(20).mean()
    d['SMA50'] = c.rolling(50).mean()
    d['EMA12'] = c.ewm(span=12).mean()
    d['EMA26'] = c.ewm(span=26).mean()
    d['DistSMA20'] = (c - d['SMA20']) / (d['SMA20'] + 1e-12)
    d['DistSMA50'] = (c - d['SMA50']) / (d['SMA50'] + 1e-12)
    d['SMAspread'] = (d['SMA20'] - d['SMA50']) / d['SMA50']
    pc = c.shift(1); ps = d['SMA20'].shift(1)
    d['CrossUp'] = ((c > d['SMA20']) & (pc <= ps)).astype(int)
    d['CrossDn'] = ((c < d['SMA20']) & (pc >= ps)).astype(int)

    delta = c.diff()
    g = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    l_ = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    d['RSI14'] = 100 - 100 / (1 + g / (l_ + 1e-12))
    d['MACD'] = d['EMA12'] - d['EMA26']
    d['MACDsignal'] = d['MACD'].ewm(span=9).mean()
    d['MACDhist'] = d['MACD'] - d['MACDsignal']

    tr = pd.concat([h - l, np.abs(h - c.shift()), np.abs(l - c.shift())], axis=1).max(axis=1)
    d['TR'] = tr
    d['ATR14'] = tr.rolling(14).mean()
    d['ATRratio'] = d['ATR14'] / (d['ATR14'].rolling(50).mean() + 1e-12)
    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    d['BBpos'] = (c - (bb_m - 2*bb_s)) / (4*bb_s + 1e-12)
    d['BBwidth'] = (4*bb_s) / (bb_m + 1e-12)

    d['Body'] = np.abs(c - o) / (d['ATR14'] + 1e-12)
    d['Uwick'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (d['ATR14'] + 1e-12)
    d['Lwick'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (d['ATR14'] + 1e-12)

    d['VolRatio'] = d['Vol'] / (d['Vol'].rolling(20).mean() + 1)

    for p in [1,2,3,5,10,20]:
        d[f'Ret{p}'] = c.pct_change(p)

    d['MR'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)

    d['Hour'] = d.index.hour.astype(int)
    d['DOW'] = d.index.dayofweek.astype(int)
    d['London'] = (d['Hour'].between(8, 15)).astype(int)
    d['Overlap'] = (d['Hour'].between(13, 15)).astype(int)

    p_dm = h.diff(); m_dm = -l.diff()
    p_dm = p_dm.where((p_dm > m_dm) & (p_dm > 0), 0)
    m_dm = m_dm.where((m_dm > p_dm) & (m_dm > 0), 0)
    pd_ = p_dm.rolling(14).mean(); md_ = m_dm.rolling(14).mean()
    pdi = 100 * pd_ / (d['ATR14'] + 1e-12); mdi = 100 * md_ / (d['ATR14'] + 1e-12)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    d['ADX'] = dx.rolling(14).mean()
    d['TrendStr'] = d['ADX'] / 50.0
    d['TrendDir'] = np.sign(d['SMAspread'])

    return d

print('Feature engineering ...')
df = add_features(df)

# ----------------------------------------------------------------------
# 4. Multi‑timeframe features
# ----------------------------------------------------------------------
c = df['Close']
df['H4SMA5'] = c.rolling(20).mean()
df['H4SMA20'] = c.rolling(80).mean()
df['H4Ret'] = c.pct_change(4)
df['D1SMA5'] = c.rolling(120).mean()
df['D1SMA20'] = c.rolling(240).mean()
df['D1Ret'] = c.pct_change(24)
df['DistH4_5'] = (c - df['H4SMA5']) / (df['H4SMA5'] + 1e-12)
df['DistH4_20'] = (c - df['H4SMA20']) / (df['H4SMA20'] + 1e-12)
df['H4cross'] = np.sign(df['H4SMA5'] - df['H4SMA20'])
df['DistD1_5'] = (c - df['D1SMA5']) / (df['D1SMA5'] + 1e-12)
df['DistD1_20'] = (c - df['D1SMA20']) / (df['D1SMA20'] + 1e-12)
df['D1cross'] = np.sign(df['D1SMA5'] - df['D1SMA20'])

df = df.iloc[240:].copy()
df.dropna(inplace=True)

FEATURES = ['DistSMA20','DistSMA50','SMAspread','CrossUp','CrossDn',
    'RSI14','MACDhist','ATRratio','BBpos','BBwidth',
    'Body','Uwick','Lwick','VolRatio','MR',
    'Ret1','Ret2','Ret3','Ret5','Ret10','Ret20',
    'TrendStr','TrendDir',
    'DistH4_5','DistH4_20','H4cross','H4Ret',
    'DistD1_5','DistD1_20','D1cross','D1Ret',
    'Hour','DOW','London','Overlap']

# ----------------------------------------------------------------------
# 5. Target creation
# ----------------------------------------------------------------------
HORIZON = 3
df['FutRet'] = df['Close'].shift(-HORIZON) / df['Close'] - 1
thr = 0.0003
df['Label'] = 0
df.loc[df['FutRet'] > thr, 'Label'] = 1
df.loc[df['FutRet'] <- thr, 'Label'] = -1
df.dropna(subset=['FutRet'], inplace=True)
df = df[df['Label'] != 0].copy()
print(f'Signal bars: {len(df)}  ({len(df)/HORIZON/24:.1f} dn.)')

# ----------------------------------------------------------------------
# 6. Walk‑forward + ensemble
# ----------------------------------------------------------------------
print('\nWalk-Forward ...')
TEST_STARTS = pd.date_range('2025-01-01', '2026-07-01', freq='2MS')
ALL = []

for i, ts in enumerate(TEST_STARTS):
    te = ts + pd.DateOffset(months=2)
    tr = df[df.index < ts]; te_d = df[(df.index >= ts) & (df.index < te)]
    if len(tr) < 800 or len(te_d) < 50: continue

    Xtr = tr[FEATURES]; ytr = tr['Label']
    preds = []
    for sd in range(5):
        m = lgb.LGBMRegressor(
            n_estimators=500+sd*50, max_depth=4+(sd%3), learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.3, reg_lambda=1.5, random_state=sd*17, verbose=-1)
        m.fit(Xtr, ytr)
        preds.append(m.predict(te_d[FEATURES]))
    raw_preds = np.mean(preds, axis=0)

    out = te_d.copy()
    out['ProbUp'] = 1 / (1 + np.exp(-4*raw_preds))
    out['ProbDn'] = 1 - out['ProbUp']

    out['Regime'] = 'neutral'
    out.loc[(out['ATRratio'] > 1.2) & (out['TrendStr'] > 0.6), 'Regime'] = 'trend_hi'
    out.loc[(out['ATRratio'] > 1.2) & (out['TrendStr'] <= 0.6), 'Regime'] = 'range_hi'
    out.loc[(out['ATRratio'] <= 1.2) & (out['TrendStr'] > 0.6), 'Regime'] = 'trend_lo'
    out.loc[(out['ATRratio'] <= 1.2) & (out['TrendStr'] <= 0.6), 'Regime'] = 'range_lo'

    ALL.append(out)
    print(f'  WF{i+1:2d}  {ts.date()} .. {te.date()}   bars={len(out):5d}')

test_all = pd.concat(ALL)
test_all = test_all[~test_all.index.duplicated(keep='first')].sort_index()
print(f'Total test bars: {len(test_all)}')

# ----------------------------------------------------------------------
# 7. Backtest
# ----------------------------------------------------------------------
CAP0 = 10000; SPREAD = 1.0; PIPV = 10
test_all['FutRet3'] = test_all['Close'].shift(-HORIZON) / test_all['Close'] - 1
test_all.dropna(subset=['FutRet3'], inplace=True)

THR = {}
for rg in test_all['Regime'].unique():
    s = test_all[test_all['Regime'] == rg]
    if len(s) < 40:
        THR[rg] = 0.55; continue
    best = 0.55; bev = -9
    for t in np.arange(0.52, 0.66, 0.01):
        lu = s['ProbUp'] > t; sd_ = s['ProbDn'] > t
        if lu.sum() < 5 or sd_.sum() < 5: continue
        wlr = ((s.loc[lu, 'FutRet3'] > 0).mean() + (s.loc[sd_, 'FutRet3'] < 0).mean()) / 2
        ev = wlr - (1 - wlr)
        if ev > bev: bev = ev; best = t
    THR[rg] = best
print('Thresholds:', {k: round(v,2) for k,v in THR.items()})

capital = CAP0; equity = [capital]; pnl = []; rg_list = []; dr_list = []; size_list = []

for idx, row in test_all.iterrows():
    rg = row['Regime']; t = THR[rg]; direction = None
    if row['ProbUp'] > t: direction = 'L'
    elif row['ProbDn'] > t: direction = 'S'
    if direction is None:
        equity.append(capital); continue

    if direction == 'L':
        rp = row['ProbUp']; pr = row['FutRet3'] / 0.0001 - SPREAD
    else:
        rp = row['ProbDn']; pr = -row['FutRet3'] / 0.0001 - SPREAD

    p = rp; q = 1 - p
    kf = max((p - q) * 0.25, 0)
    vol_f = 1 / (row['ATRratio'] + 0.5)
    size = min(kf * vol_f, 0.02)

    p_ = pr * PIPV * size * 10
    capital += p_; equity.append(capital)
    pnl.append(p_); rg_list.append(rg); dr_list.append(direction); size_list.append(size)

pnl = np.array(pnl); equity = np.array(equity); N = len(pnl)

# ----------------------------------------------------------------------
# 8. Metrics
# ----------------------------------------------------------------------
if N > 0:
    WR = (pnl > 0).mean()
    tot = (equity[-1] - CAP0) / CAP0 * 100
    MDD = ((equity / np.maximum.accumulate(equity)) - 1).min() * 100
    SR = pnl.mean() / (pnl.std() + 1e-12) * math.sqrt(252)
    GP = pnl[pnl > 0].sum(); GL = abs(pnl[pnl < 0].sum()) + 1e-12
    PF = GP / GL
    avgW = pnl[pnl > 0].mean(); avgL = pnl[pnl < 0].mean()

    print(f'\n{"="*62}')
    print('  MTF + LightGBM Ensemble (Long/Short, 3-bar)')
    print(f'{"="*62}')
    print(f'  Trades:           {N}')
    print(f'  Win Rate:         {WR:.1%}')
    print(f'  Return:           {tot:+.1f}%')
    print(f'  Max drawdown:     {MDD:.1f}%')
    print(f'  Sharpe:           {SR:.2f}')
    print(f'  Profit Factor:    {PF:.2f}')
    print(f'  Avg Win/Loss:     {avgW:+.1f} / {avgL:+.1f}')
    print(f'  Final equity:     ${equity[-1]:,.2f}')

    for nm, key in [('LONG','L'), ('SHORT','S')]:
        m = np.array(dr_list) == key
        if m.sum() > 0:
            print(f'  {nm}: n={m.sum():4d}  WR={(pnl[m]>0).mean():.1%}  P&L=${pnl[m].sum():+,.0f}')

    print('  By regime:')
    for rg in sorted(test_all.loc[pnl.index, 'Regime'].unique()):
        m = rg_list == rg
        if m.sum() < 5: continue
        print(f'    {rg:11s} n={m.sum():4d}  WR={(pnl[m]>0).mean():.1%}  P&L=${pnl[m].sum():+,.0f}')

    pd.DataFrame({'Equity': equity[1:],
                  'Drawdown_%': ((equity[1:]/np.maximum.accumulate(equity[1:]))-1)*100},
                 index=test_all.index).to_csv('equity.csv')

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(equity, color='#1f77b4', lw=1)
    ax.axhline(CAP0, color='gray', ls='--', alpha=.5)
    ax.fill_between(range(len(equity)), CAP0, equity, where=equity<CAP0, color='red', alpha=.15)
    ax.set_title(f'Equity  (WR={WR:.1%}  Ret={tot:+.1f}%  MDD={MDD:.1f}%  Sharpe={SR:.2f}  PF={PF:.2f})')
    ax.set_ylabel('$'); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig('equity_curve.png', dpi=150); plt.close()
    print('\n  Plots: equity_curve.png, equity.csv')
else:
    print('No trades')

print('\nReady.')