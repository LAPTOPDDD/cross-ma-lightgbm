import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy import signal
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
print(f"Данные: {len(df)} строк, {df.index[0]} -> {df.index[-1]}")

# ==============================================================================
# 2. VMD ДЕКОМПОЗИЦИЯ (Variational Mode Decomposition)
# ==============================================================================
def vmd_decompose(signal_data, num_modes=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """Упрощённая VMD декомпозиция на IMF компоненты"""
    T = len(signal_data)
    t = np.arange(T)
    
    # центрируем сигнал
    f_mirror = signal_data.copy()
    f_hat = np.fft.fft(f_mirror)
    f_hat_shifted = np.fft.fftshift(f_hat)
    
    u_hat = np.zeros((num_modes, T), dtype=complex)
    omega = np.zeros((num_modes, ))
    
    # инициализация
    for k in range(num_modes):
        omega[k] = 0
    
    Lam = 0
    
    u_diff = tol + np.inf
    n = 0
    
    while u_diff > tol and n < 500:
        # update first mode accumulator
        k = 0
        sum_of_hat = np.sum(u_hat - u_hat, axis=0)
        
        for k in range(num_modes):
            # update omega
            ss = f_hat_shifted - sum_of_hat - u_hat[k]
            omega[k] = np.dot(t[T//2:T], (np.abs(ss[T//2:T])**2)) / np.sum(np.abs(ss[T//2:T])**2)
            
            # update spectrum
            u_hat[k, :] = (f_hat_shifted - sum_of_hat - Lam/2) / (1 + alpha * (np.arange(T) - T//2 - omega[k])**2)
            
            # update accumulator
            sum_of_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
        
        n += 1
        u_diff = 0
        for k in range(1, num_modes):
            u_diff += np.sum(np.abs(u_hat[k, :] - u_hat[k-1, :])**2) / T
    
    # убираем зеркалирование
    u = np.zeros((num_modes, T))
    for k in range(num_modes):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
    
    return u

def simple_vmd(data, n_modes=4):
    """Быстрая эмпирическая декомпозиция через bandpass фильтры"""
    result = np.zeros((n_modes, len(data)))
    
    # нормализуем
    x = data - np.mean(data)
    
    # определяем частотные полосы
    nyquist = 0.5
    low_freqs = np.linspace(0.01, 0.45, n_modes + 1)
    
    for i in range(n_modes):
        low = low_freqs[i] / nyquist
        high = low_freqs[i+1] / nyquist
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = signal.butter(3, [low, high], btype='band')
            result[i, :] = signal.filtfilt(b, a, x)
        except:
            result[i, :] = x * (1.0 / n_modes)
    
    return result

# ==============================================================================
# 3. ГЕНЕРАЦИЯ ПРИЗНАКОВ (Multi-timeframe + VMD + Regime)
# ==============================================================================
def add_features(data, vmd_features=None):
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
    # ADX-like: strength of trend
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    data['ADX'] = dx.rolling(14).mean()
    data['Trend_Strength'] = data['ADX'] / 50.0  # нормализованный
    data['Trend_Direction'] = np.sign(data['SMA_Spread'])
    
    # Bollinger Bandwidth как меру волатильности
    data['BB_Volatility'] = data['BB_Width'] / (data['BB_Width'].rolling(50).mean() + 1e-10)
    
    return data

# ==============================================================================
# 4. МУЛЬТИ-ТАЙМФРЕЙМ ПРИЗНАКИ
# ==============================================================================
def add_mtf_features(data):
    """Добавляем признаки с H4 и D1 таймфреймов через ресемплинг"""
    c = data['Close']
    
    # H4 (ресемплинг 4 свечи)
    h4 = data.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    h4['H4_SMA_5'] = h4['Close'].rolling(5).mean()
    h4['H4_SMA_20'] = h4['Close'].rolling(20).mean()
    h4['H4_Ret_1'] = h4['Close'].pct_change(1)
    h4['H4_Ret_3'] = h4['Close'].pct_change(3)
    
    # выравниваем по времени
    h4_features = h4[['H4_SMA_5', 'H4_SMA_20', 'H4_Ret_1', 'H4_Ret_3']].copy()
    h4_features = h4_features.resample('1H').ffill()
    
    # D1 (дневной)
    d1 = data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    d1['D1_SMA_5'] = d1['Close'].rolling(5).mean()
    d1['D1_SMA_20'] = d1['Close'].rolling(20).mean()
    d1['D1_Ret_1'] = d1['Close'].pct_change(1)
    
    d1_features = d1[['D1_SMA_5', 'D1_SMA_20', 'D1_Ret_1']].copy()
    d1_features = d1_features.resample('1H').ffill()
    
    # мерджим
    data = data.join(h4_features, how='left')
    data = data.join(d1_features, how='left')
    
    # относительные расстояния
    data['Dist_H4_SMA5'] = (data['Close'] - data['H4_SMA_5']) / (data['H4_SMA_5'] + 1e-10)
    data['Dist_H4_SMA20'] = (data['Close'] - data['H4_SMA_20']) / (data['H4_SMA_20'] + 1e-10)
    data['H4_SMA_Cross'] = np.sign(data['H4_SMA_5'] - data['H4_SMA_20'])
    
    data['Dist_D1_SMA5'] = (data['Close'] - data['D1_SMA_5']) / (data['D1_SMA_5'] + 1e-10)
    data['Dist_D1_SMA20'] = (data['Close'] - data['D1_SMA_20']) / (data['D1_SMA_20'] + 1e-10)
    data['D1_SMA_Cross'] = np.sign(data['D1_SMA_5'] - data['D1_SMA_20'])
    
    return data

# ==============================================================================
# 5. VMD ПРИЗНАКИ
# ==============================================================================
def add_vmd_features(data, n_modes=4):
    """Добавляем признаки на основе VMD декомпозиции цены"""
    close = data['Close'].values
    
    # VMD декомпозиция
    try:
        imfs = simple_vmd(close, n_modes)
    except:
        return data, None
    
    imf_names = []
    for i in range(n_modes):
        name = f'IMF_{i}'
        data[name] = imfs[i]
        imf_names.append(name)
        
        # признаки для каждой IMF
        data[f'IMF_{i}_ret1'] = pd.Series(imfs[i]).pct_change(1).values
        data[f'IMF_{i}_ret5'] = pd.Series(imfs[i]).pct_change(5).values
        data[f'IMF_{i}_mean'] = pd.Series(imfs[i]).rolling(10).mean().values
        data[f'IMF_{i}_std'] = pd.Series(imfs[i]).rolling(10).std().values
    
    # Low-freq IMF = тренд, High-freq IMF = шум
    data['VMD_Trend'] = data['IMF_0'] + data['IMF_1']  # низкочастотные
    data['VMD_Noise'] = data['IMF_2'] + data['IMF_3']  # высокочастотные
    data['VMD_SNR'] = data['VMD_Trend'] / (data['VMD_Noise'].abs() + 1e-10)
    
    return data, imf_names

# ==============================================================================
# 6. ОСНОВНОЙ ПАЙПЛАЙН
# ==============================================================================
print("\n--- Feature Engineering ---")
df = add_features(df)
df = add_mtf_features(df)
df, vmd_names = add_vmd_features(df, n_modes=4)

# Target: следующая свеча вверх?
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# Заполняем NaNMean
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)
df.dropna(inplace=True)
print(f"После признаков: {len(df)} строк")

# ==============================================================================
# 7. WALK-FORWARD С ДИНАМИЧЕСКИМ ENSEMBLE
# ==============================================================================
feature_cols = [
    'Dist_SMA20', 'Dist_SMA50', 'SMA_Spread',
    'Cross_Up', 'Cross_Down',
    'RSI_14', 'MACD_Hist', 'ATR_Ratio', 'BB_Pos', 'BB_Width',
    'Body', 'UpperWick', 'LowerWick',
    'Vol_Ratio', 'MR_Score',
    'Ret_1', 'Ret_2', 'Ret_3', 'Ret_5', 'Ret_10',
    'Trend_Strength', 'Trend_Direction', 'BB_Volatility',
    'Dist_H4_SMA5', 'Dist_H4_SMA20', 'H4_SMA_Cross',
    'Dist_D1_SMA5', 'Dist_D1_SMA20', 'D1_SMA_Cross',
    'Hour', 'DayOfWeek', 'London', 'Overlap',
]

# Добавляем VMD признаки если есть
if vmd_names:
    for name in ['VMD_Trend', 'VMD_Noise', 'VMD_SNR']:
        if name in df.columns:
            feature_cols.append(name)

print(f"Признаков: {len(feature_cols)}")

# Walk-forward: каждые 2 месяца переобучаем
split_start = pd.Timestamp('2024-06-01')
test_months = pd.date_range('2025-01-01', '2026-07-01', freq='2MS')

all_test_results = []

for i, test_start in enumerate(test_months):
    test_end = test_start + pd.DateOffset(months=2)
    
    train_data = df[df.index < test_start].copy()
    test_data = df[(df.index >= test_start) & (df.index < test_end)].copy()
    
    if len(train_data) < 500 or len(test_data) < 50:
        continue
    
    X_tr = train_data[feature_cols].fillna(0)
    y_tr = train_data['Target']
    X_te = test_data[feature_cols].fillna(0)
    y_te = test_data['Target']
    
    # Ансамбль из 5 моделей с разными гиперпараметрами
    configs = [
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.04, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 350, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.65, 'colsample_bytree': 0.85},
    ]
    
    ensemble_probs = []
    for cfg in configs:
        m = lgb.LGBMClassifier(
            **cfg,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        m.fit(X_tr, y_tr)
        ensemble_probs.append(m.predict_proba(X_te)[:, 1])
    
    # Среднее ансамбля
    avg_probs = np.mean(ensemble_probs, axis=0)
    
    result = test_data.copy()
    result['Prob'] = avg_probs
    result['Target'] = y_te.values
    
    # Regime: определяем по ATR_Ratio и ADX
    result['Regime'] = 'neutral'
    result.loc[(result['ATR_Ratio'] > 1.2) & (result['Trend_Strength'] > 0.6), 'Regime'] = 'trend_high_vol'
    result.loc[(result['ATR_Ratio'] > 1.2) & (result['Trend_Strength'] <= 0.6), 'Regime'] = 'range_high_vol'
    result.loc[(result['ATR_Ratio'] <= 1.2) & (result['Trend_Strength'] > 0.6), 'Regime'] = 'trend_low_vol'
    result.loc[(result['ATR_Ratio'] <= 1.2) & (result['Trend_Strength'] <= 0.6), 'Regime'] = 'range_low_vol'
    
    all_test_results.append(result)
    print(f"  Window {i+1}: train<{test_start.date()}, test={test_start.date()}-{test_end.date()}, "
          f"acc={(avg_probs.round() == y_te.values).mean():.3f}")

# Объединяем
test_all = pd.concat(all_test_results)
test_all = test_all[~test_all.index.duplicated(keep='first')]
test_all.sort_index(inplace=True)
print(f"\nВсего тестовых: {len(test_all)}")

# ==============================================================================
# 8. БЭКТЕСТ С ДИНАМИЧЕСКИМ WEIGHING ПО РЕЖИМУ
# ==============================================================================
capital_start = 10000
spread_pips = 1.0
pip_value = 10

# Actual return (next bar)
test_all['Next_Return'] = (test_all['Close'].shift(-1) - test_all['Close']) / 0.0001
test_all.dropna(subset=['Next_Return'], inplace=True)

# Динамическое взвешивание по региону
# В тренде: доверяем direction-based signals
# В боковике: доверяем mean-reversion signals

# Определяем порог для каждого режима оптимальным образом на historical данных
print("\n--- Optimizing regime-specific thresholds ---")

regimes = test_all['Regime'].unique()
regime_params = {}

for regime in regimes:
    regime_data = test_all[test_all['Regime'] == regime]
    if len(regime_data) < 30:
        regime_params[regime] = {'threshold': 0.55, 'direction': 'both'}
        continue
    
    best_thr = 0.55
    best_edge = -1
    
    for thr in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
        # Long
        long_mask = regime_data['Prob'] > thr
        short_mask = regime_data['Prob'] < (1 - thr)
        
        if long_mask.sum() + short_mask.sum() < 10:
            continue
        
        long_wr = regime_data.loc[long_mask, 'Target'].mean() if long_mask.sum() > 0 else 0
        short_wr = (1 - regime_data.loc[short_mask, 'Target'].mean()) if short_mask.sum() > 0 else 0
        
        total_trades = long_mask.sum() + short_mask.sum()
        total_correct = regime_data.loc[long_mask, 'Target'].sum() + (1 - regime_data.loc[short_mask, 'Target']).sum()
        edge = total_correct / max(total_trades, 1) - 0.5
        
        if edge > best_edge and total_trades >= 5:
            best_edge = edge
            best_thr = thr
    
    regime_params[regime] = {'threshold': best_thr}
    print(f"  {regime}: threshold={best_thr:.2f}, samples={len(regime_data)}")

# Бэктест
capital = capital_start
equity = [capital]
trade_pnls = []
trade_details = []

for idx, row in test_all.iterrows():
    prob = row['Prob']
    regime = row['Regime']
    thr = regime_params.get(regime, {}).get('threshold', 0.55)
    
    # Динамический размер позиции по уверенности
    confidence = abs(prob - 0.5) * 2  # 0..1
    
    if prob > thr:
        # Long
        direction = 1
        ret_pips = row['Next_Return'] - spread_pips
    elif prob < (1 - thr):
        # Short
        direction = -1
        ret_pips = -row['Next_Return'] - spread_pips
    else:
        equity.append(capital)
        continue
    
    # Position size: Kelly-like, proportional to confidence
    pos_size = min(confidence * 0.5, 0.15)  # max 15%
    pos_size = max(pos_size, 0.02)  # min 2%
    
    pnl = ret_pips * pip_value * pos_size * 10  # scaled
    capital += pnl
    trade_pnls.append(pnl)
    trade_details.append({
        'time': idx, 'direction': 'Long' if direction == 1 else 'Short',
        'prob': prob, 'regime': regime, 'thr': thr, 'pnl': pnl, 'ret_pips': ret_pips,
    })
    equity.append(capital)

trade_pnls = np.array(trade_pnls)
equity = np.array(equity)

# ==============================================================================
# 9. МЕТРИКИ
# ==============================================================================
n = len(trade_pnls)
wins = (trade_pnls > 0).sum()
wr = wins / n if n > 0 else 0
total_ret = (equity[-1] - capital_start) / capital_start * 100
max_dd = ((equity / np.maximum.accumulate(equity)) - 1).min() * 100
sharpe = trade_pnls.mean() / (trade_pnls.std() + 1e-10) * np.sqrt(252)
gross_profit = trade_pnls[trade_pnls > 0].sum() if (trade_pnls > 0).any() else 0
gross_loss = abs(trade_pnls[trade_pnls < 0].sum()) if (trade_pnls < 0).any() else 1e-10
pf = gross_profit / gross_loss
avg_win = trade_pnls[trade_pnls > 0].mean() if (trade_pnls > 0).any() else 0
avg_loss = trade_pnls[trade_pnls < 0].mean() if (trade_pnls < 0).any() else 0

# Profit Factor by regime
print("\n--- Performance by Regime ---")
for regime in regimes:
    mask = [i for i, d in enumerate(trade_details) if d['regime'] == regime]
    if len(mask) < 5:
        continue
    rp = trade_pnls[mask]
    r_wr = (rp > 0).sum() / len(rp)
    r_ret = rp.sum()
    print(f"  {regime:20s}: trades={len(rp):3d}, WR={r_wr:.1%}, P&L=${r_ret:+.0f}")

print(f"\n{'='*60}")
print(f"РЕЗУЛЬТАТЫ (Multi-timeframe + VMD + Regime-dynamic Ensemble)")
print(f"{'='*60}")
print(f"  Всего сделок: {n}")
print(f"  Win Rate: {wr:.1%}")
print(f"  Доходность: {total_ret:+.1f}%")
print(f"  Макс. просадка: {max_dd:.1f}%")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Средний выигрыш: ${avg_win:.1f}")
print(f"  Средний проигрыш: ${avg_loss:.1f}")
print(f"  Финальный капитал: ${equity[-1]:,.2f}")

# ==============================================================================
# 10. ВИЗУАЛИЗАЦИЯ
# ==============================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle(f'MTF + VMD + Regime Ensemble (Sharpe={sharpe:.2f}, PF={pf:.2f})', fontsize=14)

# Equity
axes[0].plot(equity, color='blue', linewidth=1)
axes[0].axhline(y=capital_start, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('Equity Curve')
axes[0].set_ylabel('$')
axes[0].grid(True, alpha=0.3)

# Probability distribution
axes[1].hist(test_all['Prob'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[1].set_title('Probability Distribution')
axes[1].grid(True, alpha=0.3)

# P&L distribution
axes[2].hist(trade_pnls, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
axes[2].axvline(x=0, color='red', linestyle='--')
axes[2].set_title(f'Trade P&L (WR={wr:.1%}, PF={pf:.2f})')
axes[2].set_xlabel('P&L ($)')
axes[2].grid(True, alpha=0.3)

# Cumulative P&L by regime
colors = {'trend_high_vol': 'green', 'trend_low_vol': 'lime', 
          'range_high_vol': 'orange', 'range_low_vol': 'gray', 'neutral': 'blue'}
cum_pnl = 0
for regime in regimes:
    mask = [i for i, d in enumerate(trade_details) if d['regime'] == regime]
    if len(mask) < 2:
        continue
    rp = trade_pnls[mask]
    cp = np.cumsum(rp)
    axes[3].plot(cp, label=regime, color=colors.get(regime, 'gray'), linewidth=1)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[3].set_title('Cumulative P&L by Regime')
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_v6_mtf_vmd_regime.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафик: backtest_v6_mtf_vmd_regime.png")
