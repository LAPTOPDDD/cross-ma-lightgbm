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
# 2. ПРИЗНАКИ ДЛЯ КАЖДОГО ТФ
# =============================================================================
def add_tf_features(data, prefix):
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
# 3. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# =============================================================================
def test_symbol(symbol, fixed_lots=0.05, data_dir='mt5_downloaded_data', output_dir='v30_final_results'):
    print(f"\n{'='*60}")
    print(f"ТЕСТИРУЕМ: {symbol} | Размер позиции: {fixed_lots} лота")
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

    h1_feat = add_tf_features(h1, 'H1')
    h4_feat = add_tf_features(h4, 'H4')
    d1_feat = add_tf_features(d1, 'D1')

    levels = pd.DataFrame(index=d1.index)
    levels['D1_PrevHigh'] = d1['High'].shift(1)
    levels['D1_PrevLow'] = d1['Low'].shift(1)
    levels['D1_PrevClose'] = d1['Close'].shift(1)

    def merge_past_only(h1_idx, higher_feat):
        return higher_feat.reindex(h1_idx, method='ffill')

    df = h1_feat.copy()
    h4_merged = merge_past_only(df.index, h4_feat)
    d1_merged = merge_past_only(df.index, d1_feat)
    levels_merged = merge_past_only(df.index, levels)

    h4_merged.columns = [c + '_H4' for c in h4_merged.columns]
    d1_merged.columns = [c + '_D1' for c in d1_merged.columns]
    levels_merged.columns = [c + '_LVL' for c in levels_merged.columns]

    df = df.join(h4_merged, how='left')
    df = df.join(d1_merged, how='left')
    df = df.join(levels_merged, how='left')

    df['Dist_D1_PrevHigh'] = (h1['Close'].reindex(df.index) - df['D1_PrevHigh_LVL']) / df['D1_PrevHigh_LVL']
    df['Dist_D1_PrevLow'] = (h1['Close'].reindex(df.index) - df['D1_PrevLow_LVL']) / df['D1_PrevLow_LVL']

    df['H1_Hour'] = df.index.hour
    df['H1_DayOfWeek'] = df.index.dayofweek
    df['H1_London'] = df['H1_Hour'].between(8, 15).astype(int)
    df['H1_Overlap'] = df['H1_Hour'].between(13, 15).astype(int)

    df = df.iloc[250:].copy()

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

    df['Regime'] = df.apply(regime_of, axis=1)
    regimes = ['trend_hi', 'range_hi', 'trend_lo', 'range_lo']

    df['Target'] = (h1['Close'].reindex(df.index).shift(-1) > h1['Close'].reindex(df.index)).astype(int)
    df.dropna(subset=['Target'], inplace=True)

    exclude = ['Target', 'Regime']
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"  Данные: {len(df)} строк, признаков: {len(feature_cols)}")

    # Walk-forward
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

    if len(pred_by_model[0]) == 0:
        print(f"  Пропуск: недостаточно данных для walk-forward")
        return None

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

    # Бэктест
    CAPITAL_START = 10000
    SPREAD_PIPS = 0.5  # более реалистичный спред
    COMMISSION_PER_LOT = 6  # $6 за лот
    PIP_VALUE = 10
    ATR_SL_MULT = 2.0
    ATR_TP_MULT = 4.0
    MAX_HOLD_BARS = 36
    SWAP_PER_LOT_PER_HOUR = 0.05

    def get_pip_value(close_price):
        if symbol in ['XAUUSD', 'XAGUSD']:
            return 0.1
        elif 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001

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
    pip_val = get_pip_value(h1['Close'].iloc[-1])

    for idx, row in test_all.iterrows():
        if position != 0:
            unrealized_pips = (h1.loc[idx, 'Close'] - entry_price) / pip_val
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
                exit_pips = (sl - entry_price) / pip_val - SPREAD_PIPS
                exit_reason = 'sl'
            elif high >= tp:
                exit_pips = (tp - entry_price) / pip_val - SPREAD_PIPS
                exit_reason = 'tp'
            elif bars_in_trade >= MAX_HOLD_BARS:
                exit_pips = (h1.loc[idx, 'Close'] - entry_price) / pip_val - SPREAD_PIPS
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
                'swap': swap
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
            if row['Prob'] <= FIXED_THR:
                continue

            entry_price = h1.loc[idx, 'Close']
            entry_time = idx
            sl = entry_price - row['H1_ATR_14'] * ATR_SL_MULT
            tp = entry_price + row['H1_ATR_14'] * ATR_TP_MULT
            pos_size = fixed_lots
            position = 1
            bars_in_trade = 0

    if position != 0:
        last_idx = test_all.index[-1]
        last_close = h1.loc[last_idx, 'Close']
        exit_pips = (last_close - entry_price) / pip_val - SPREAD_PIPS
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
    eq_df = pd.DataFrame({'Equity': equity, 'Drawdown_%': drawdown}, index=test_all.index)
    eq_df.to_csv(os.path.join(output_dir, f'equity_{symbol}.csv'))

    trades_df = pd.DataFrame(trade_meta)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(output_dir, f'trades_{symbol}.csv'), index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    axes[0].plot(eq_df.index, eq_df['Equity'], color='teal', linewidth=1.2)
    axes[0].axhline(y=CAPITAL_START, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title(f'v30 {symbol} | Ret={total_return:+.1f}% | MaxDD={max_dd:.1f}% | Sharpe={sharpe:.2f} | Trades={n_trades}', fontsize=13)
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
    plt.savefig(os.path.join(output_dir, f'backtest_{symbol}.png'), dpi=150, bbox_inches='tight')
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
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'XAUUSD', 'XAGUSD']
FIXED_LOTS = 0.05

results = []
for symbol in SYMBOLS:
    res = test_symbol(symbol, fixed_lots=FIXED_LOTS)
    if res:
        results.append(res)

# Сводная таблица
if results:
    summary = pd.DataFrame(results)
    summary = summary.sort_values('sharpe', ascending=False)
    summary.to_csv('v30_final_results/summary.csv', index=False)

    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА (отсортирована по Sharpe)")
    print(f"{'='*80}")
    print(summary.to_string(index=False))
    print(f"\nСохранено: v30_final_results/summary.csv")

    # Общий график сравнения всех активов
    fig, ax = plt.subplots(figsize=(14, 8))
    for symbol in SYMBOLS:
        eq_path = os.path.join('v30_final_results', f'equity_{symbol}.csv')
        if os.path.exists(eq_path):
            eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
            ax.plot(eq.index, eq['Equity'], label=symbol, linewidth=1.2)

    ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'v30 Universal Strategy Comparison | Fixed size: {FIXED_LOTS} lot', fontsize=14)
    ax.set_ylabel('Capital, $')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('v30_final_results/comparison_all_assets.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранено: v30_final_results/comparison_all_assets.png")
