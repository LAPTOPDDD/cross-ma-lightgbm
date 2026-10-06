import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ЗАГРУЗКА MT5 CSV
# =============================================================================
def load_mt5_csv(path):
    df = pd.read_csv(path)
    df['DateTime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
    return df

# =============================================================================
# 2. ПРИЗНАКИ
# =============================================================================
def add_tf_features(data, prefix):
    """Добавляет признаки для одного ТФ. data — DataFrame с OHLCV."""
    c, h, l, o = data['Close'], data['High'], data['Low'], data['Open']
    out = pd.DataFrame(index=data.index)

    out[f'{prefix}_Return_1'] = c.pct_change(1)
    out[f'{prefix}_Return_4'] = c.pct_change(4)
    out[f'{prefix}_Dist_SMA20'] = (c - c.rolling(20).mean()) / c.rolling(20).mean()
    out[f'{prefix}_Dist_SMA50'] = (c - c.rolling(50).mean()) / c.rolling(50).mean()

    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    out[f'{prefix}_RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    out[f'{prefix}_ATR_14'] = tr.rolling(14).mean()
    out[f'{prefix}_ATR_Ratio'] = out[f'{prefix}_ATR_14'] / (out[f'{prefix}_ATR_14'].rolling(50).mean() + 1e-10)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out[f'{prefix}_BB_Pos'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)

    out[f'{prefix}_MR_Score'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)

    return out

# =============================================================================
# 3. ONLINE WALK-FORWARD
# =============================================================================
def online_test_symbol(symbol, data_dir='mt5_downloaded_data', output_dir='v31_online_results'):
    print(f"\n{'='*60}")
    print(f"ONLINE ТЕСТ: {symbol}")
    print(f"{'='*60}")

    h1_path = os.path.join(data_dir, f'{symbol}_H1_20240101_20260909.csv')
    h4_path = os.path.join(data_dir, f'{symbol}_H4_20240101_20260909.csv')
    d1_path = os.path.join(data_dir, f'{symbol}_D1_20240101_20260909.csv')

    if not all(os.path.exists(p) for p in [h1_path, h4_path, d1_path]):
        print(f"  Пропуск: недостаточно файлов")
        return None

    h1 = load_mt5_csv(h1_path)
    h4 = load_mt5_csv(h4_path)
    d1 = load_mt5_csv(d1_path)

    print(f"  H1: {len(h1)} | H4: {len(h4)} | D1: {len(d1)}")

    # Периоды
    train_end = pd.Timestamp('2025-03-01')
    test_start = pd.Timestamp('2025-03-01')

    # Разбиваем данные
    h1_train = h1[h1.index < train_end]
    h1_test = h1[h1.index >= test_start]
    h4_train = h4[h4.index < train_end]
    d1_train = d1[d1.index < train_end]

    # Минимум данных для обучения
    if len(h1_train) < 1000:
        print(f"  Пропуск: недостаточно тренировочных данных")
        return None

    # =============================================================================
    # 3.1 Обучаем модель один раз на истории
    # =============================================================================
    print(f"  Обучаем модель на {len(h1_train)} H1 барах...")

    # Строим признаки на тренировочных данных
    h1_feat_train = add_tf_features(h1_train, 'H1')
    h4_feat_train = add_tf_features(h4_train, 'H4')
    d1_feat_train = add_tf_features(d1_train, 'D1')

    levels_train = pd.DataFrame(index=d1_train.index)
    levels_train['D1_PrevHigh'] = d1_train['High'].shift(1)
    levels_train['D1_PrevLow'] = d1_train['Low'].shift(1)

    df_train = h1_feat_train.copy()
    h4_merged = h4_feat_train.reindex(df_train.index, method='ffill')
    d1_merged = d1_feat_train.reindex(df_train.index, method='ffill')
    levels_merged = levels_train.reindex(df_train.index, method='ffill')

    h4_merged.columns = [c + '_H4' for c in h4_merged.columns]
    d1_merged.columns = [c + '_D1' for c in d1_merged.columns]
    levels_merged.columns = [c + '_LVL' for c in levels_merged.columns]

    df_train = df_train.join(h4_merged, how='left')
    df_train = df_train.join(d1_merged, how='left')
    df_train = df_train.join(levels_merged, how='left')

    df_train['Dist_D1_PrevHigh'] = (h1_train['Close'].reindex(df_train.index) - df_train['D1_PrevHigh_LVL']) / df_train['D1_PrevHigh_LVL']
    df_train['Dist_D1_PrevLow'] = (h1_train['Close'].reindex(df_train.index) - df_train['D1_PrevLow_LVL']) / df_train['D1_PrevLow_LVL']
    df_train['H1_Hour'] = df_train.index.hour
    df_train['H1_DayOfWeek'] = df_train.index.dayofweek
    df_train['H1_London'] = df_train['H1_Hour'].between(8, 15).astype(int)
    df_train['H1_Overlap'] = df_train['H1_Hour'].between(13, 15).astype(int)

    # Target: следующий бар выше текущего
    df_train['Target'] = (h1_train['Close'].shift(-1) > h1_train['Close']).astype(int)
    df_train = df_train.iloc[250:].copy()  # убираем NaN от скользящих окон
    df_train.dropna(subset=['Target'], inplace=True)

    # Режимы
    def regime_of(r):
        if (r['H1_ATR_Ratio'] > 1.2) and (r['H1_RSI_14'] > 60):
            return 'trend_hi'
        elif r['H1_ATR_Ratio'] > 1.2:
            return 'range_hi'
        elif r['H1_RSI_14'] > 60:
            return 'trend_lo'
        else:
            return 'range_lo'

    df_train['Regime'] = df_train.apply(regime_of, axis=1)

    exclude = ['Target', 'Regime']
    feature_cols = [c for c in df_train.columns if c not in exclude]

    ALLOWED = ['range_lo', 'range_hi']
    FIXED_THR = 0.60

    # Обучаем ансамбль из 7 моделей
    N_MODELS = 7
    models = []
    for seed in range(N_MODELS):
        m = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=100,
            reg_alpha=0.5, reg_lambda=2.0, random_state=seed * 7, verbose=-1
        )
        m.fit(df_train[feature_cols], df_train['Target'])
        models.append(m)

    print(f"  Модель обучена. Признаков: {len(feature_cols)}")

    # =============================================================================
    # 3.2 ONLINE ТЕСТ: подаём бары по одному
    # =============================================================================
    print(f"  Запускаем online тест на {len(h1_test)} барах...")

    # Инициализация истории для пересчёта признаков
    # Берём последние 300 баров из тренировки + будем добавлять тестовые бары
    h1_history = h1_train.copy()
    h4_history = h4_train.copy()
    d1_history = d1_train.copy()

    # Бэктест
    CAPITAL_START = 10000
    SPREAD_PIPS = 0.5
    COMMISSION_PER_LOT = 6
    PIP_VALUE = 10
    ATR_SL_MULT = 2.0
    ATR_TP_MULT = 4.0
    MAX_HOLD_BARS = 36
    SWAP_PER_LOT_PER_HOUR = 0.05
    FIXED_LOTS = 0.05

    def get_pip_value(close_price):
        if symbol in ['XAUUSD', 'XAGUSD']:
            return 0.1
        elif 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001

    pip_val = get_pip_value(h1_train['Close'].iloc[-1])

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

    # Для каждого тестового бара
    test_bars = h1_test.index
    n_test = len(test_bars)

    for i, idx in enumerate(test_bars):
        if i % 500 == 0:
            print(f"    Обработано {i}/{n_test} баров...")

        # Добавляем текущий бар в историю
        h1_history = pd.concat([h1_history, h1_test.loc[[idx]]])

        # Обновляем H4 и D1 истории, если текущий бар — это закрытие нового H4/D1 бара
        # H4 бары закрываются в 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        # D1 бары закрываются в 00:00
        current_time = idx
        if current_time.hour in [0, 4, 8, 12, 16, 20] and current_time.minute == 0:
            # Это потенциальное закрытие H4 бара
            # Но нам нужен уже сформированный H4 бар. Проверим, есть ли он в h4_test
            if idx in h4.index and idx not in h4_history.index:
                h4_history = pd.concat([h4_history, h4.loc[[idx]]])
        if current_time.hour == 0 and current_time.minute == 0:
            # Закрытие D1 бара
            if idx in d1.index and idx not in d1_history.index:
                d1_history = pd.concat([d1_history, d1.loc[[idx]]])

        # Пересчитываем признаки на истории (только прошлое!)
        h1_feat_hist = add_tf_features(h1_history, 'H1')
        h4_feat_hist = add_tf_features(h4_history, 'H4')
        d1_feat_hist = add_tf_features(d1_history, 'D1')

        levels_hist = pd.DataFrame(index=d1_history.index)
        levels_hist['D1_PrevHigh'] = d1_history['High'].shift(1)
        levels_hist['D1_PrevLow'] = d1_history['Low'].shift(1)

        df_hist = h1_feat_hist.copy()
        h4_merged = h4_feat_hist.reindex(df_hist.index, method='ffill')
        d1_merged = d1_feat_hist.reindex(df_hist.index, method='ffill')
        levels_merged = levels_hist.reindex(df_hist.index, method='ffill')

        h4_merged.columns = [c + '_H4' for c in h4_merged.columns]
        d1_merged.columns = [c + '_D1' for c in d1_merged.columns]
        levels_merged.columns = [c + '_LVL' for c in levels_merged.columns]

        df_hist = df_hist.join(h4_merged, how='left')
        df_hist = df_hist.join(d1_merged, how='left')
        df_hist = df_hist.join(levels_merged, how='left')

        df_hist['Dist_D1_PrevHigh'] = (h1_history['Close'].reindex(df_hist.index) - df_hist['D1_PrevHigh_LVL']) / df_hist['D1_PrevHigh_LVL']
        df_hist['Dist_D1_PrevLow'] = (h1_history['Close'].reindex(df_hist.index) - df_hist['D1_PrevLow_LVL']) / df_hist['D1_PrevLow_LVL']
        df_hist['H1_Hour'] = df_hist.index.hour
        df_hist['H1_DayOfWeek'] = df_hist.index.dayofweek
        df_hist['H1_London'] = df_hist['H1_Hour'].between(8, 15).astype(int)
        df_hist['H1_Overlap'] = df_hist['H1_Hour'].between(13, 15).astype(int)

        # Берём признаки текущего бара
        current_features = df_hist.loc[[idx], feature_cols]

        # Предсказание ансамбля
        probs = []
        for m in models:
            probs.append(m.predict_proba(current_features)[0, 1])
        current_prob = np.mean(probs)

        # Текущий режим
        current_regime = regime_of(df_hist.loc[idx])

        # Обновляем эквити
        if position != 0:
            unrealized_pips = (h1_test.loc[idx, 'Close'] - entry_price) / pip_val
            unrealized = unrealized_pips * PIP_VALUE * pos_size
            current_equity = capital + unrealized
        else:
            current_equity = capital
        equity.append(current_equity)

        # Проверяем закрытие позиции
        if position != 0:
            bars_in_trade += 1
            high = h1_test.loc[idx, 'High']
            low = h1_test.loc[idx, 'Low']
            exit_reason = None

            if low <= sl:
                exit_pips = (sl - entry_price) / pip_val - SPREAD_PIPS
                exit_reason = 'sl'
            elif high >= tp:
                exit_pips = (tp - entry_price) / pip_val - SPREAD_PIPS
                exit_reason = 'tp'
            elif bars_in_trade >= MAX_HOLD_BARS:
                exit_pips = (h1_test.loc[idx, 'Close'] - entry_price) / pip_val - SPREAD_PIPS
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
                'pnl': pnl, 'reason': exit_reason, 'bars': bars_in_trade,
                'swap': swap, 'prob': current_prob
            })
            position = 0
            entry_price = 0
            entry_time = None
            pos_size = 0
            sl = 0
            tp = 0
            bars_in_trade = 0

        # Проверяем открытие позиции
        if position == 0:
            if current_regime not in ALLOWED:
                continue
            if current_prob <= FIXED_THR:
                continue

            entry_price = h1_test.loc[idx, 'Close']
            entry_time = idx
            atr = df_hist.loc[idx, 'H1_ATR_14']
            sl = entry_price - atr * ATR_SL_MULT
            tp = entry_price + atr * ATR_TP_MULT
            pos_size = FIXED_LOTS
            position = 1
            bars_in_trade = 0

    # Закрываем в конце
    if position != 0:
        last_idx = test_bars[-1]
        last_close = h1_test.loc[last_idx, 'Close']
        exit_pips = (last_close - entry_price) / pip_val - SPREAD_PIPS
        swap = SWAP_PER_LOT_PER_HOUR * pos_size * bars_in_trade
        commission = COMMISSION_PER_LOT * pos_size
        pnl = exit_pips * PIP_VALUE * pos_size - commission - swap
        capital += pnl
        trade_pnls.append(pnl)
        trade_meta.append({
            'entry': entry_time, 'exit': last_idx, 'size': pos_size,
            'pnl': pnl, 'reason': 'end', 'bars': bars_in_trade,
            'swap': swap, 'prob': current_prob
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

    print(f"\n  === РЕЗУЛЬТАТЫ ONLINE ТЕСТА {symbol} ===")
    print(f"  Сделок: {n_trades}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Доходность: {total_return:+.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Сред. удержание: {avg_bars:.1f} баров")
    print(f"  Итоговый капитал: ${equity[-1]:,.2f}")

    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    eq_df = pd.DataFrame({'Equity': equity, 'Drawdown_%': drawdown}, index=test_bars)
    eq_df.to_csv(os.path.join(output_dir, f'equity_{symbol}_online.csv'))

    trades_df = pd.DataFrame(trade_meta)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(output_dir, f'trades_{symbol}_online.csv'), index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    axes[0].plot(eq_df.index, eq_df['Equity'], color='teal', linewidth=1.2)
    axes[0].axhline(y=CAPITAL_START, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title(f'v31 Online {symbol} | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | Trades={n_trades}', fontsize=13)
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
    plt.savefig(os.path.join(output_dir, f'backtest_{symbol}_online.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'symbol': symbol,
        'trades': n_trades,
        'win_rate': win_rate,
        'return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'profit_factor': profit_factor,
        'final_capital': equity[-1],
        'avg_bars': avg_bars
    }

# =============================================================================
# 4. ЗАПУСК ПО ВСЕМ АКТИВАМ
# =============================================================================
#SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'XAUUSD', 'XAGUSD']
SYMBOLS = ['XAUUSD']

results = []
for symbol in SYMBOLS:
    res = online_test_symbol(symbol)
    if res:
        results.append(res)

if results:
    summary = pd.DataFrame(results)
    summary = summary.sort_values('sharpe', ascending=False)
    os.makedirs('v31_online_results', exist_ok=True)
    summary.to_csv('v31_online_results/summary.csv', index=False)

    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА ONLINE ТЕСТА")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    # Сравнительный график
    fig, ax = plt.subplots(figsize=(14, 8))
    for symbol in SYMBOLS:
        eq_path = os.path.join('v31_online_results', f'equity_{symbol}_online.csv')
        if os.path.exists(eq_path):
            eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
            ax.plot(eq.index, eq['Equity'], label=symbol, linewidth=1.2)

    ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('v31 Online Walk-Forward Comparison', fontsize=14)
    ax.set_ylabel('Capital, $')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('v31_online_results/comparison_all_assets_online.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nСохранено: v31_online_results/comparison_all_assets_online.png")
