import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0f172a'
plt.rcParams['axes.facecolor'] = '#1e293b'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'


class EURUSDMLSystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.features = None
        self.model = None
        self.feature_names = None
        self.splits = None

    def load_data(self):
        """Загрузка и парсинг CSV данных"""
        print("📊 Загрузка данных...")

        self.data = pd.read_csv(
            self.csv_path,
            sep='\t',
            header=0,
            names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']
        )

        self.data['DATETIME'] = pd.to_datetime(
            self.data['DATE'].astype(str) + ' ' + self.data['TIME'].astype(str),
            format='%Y.%m.%d %H:%M:%S'
        )

        self.data = self.data.sort_values('DATETIME').reset_index(drop=True)

        print(f"✅ Загружено {len(self.data)} баров")
        print(f"📅 Период: {self.data['DATETIME'].min()} → {self.data['DATETIME'].max()}")

        return self.data

    def calculate_features(self):
        """Расчет технических индикаторов"""
        print("\n🔧 Расчет признаков...")

        df = self.data.copy()

        # SMA
        for period in [10, 20, 50, 100]:
            df[f'SMA_{period}'] = df['CLOSE'].rolling(window=period).mean()
            df[f'SMA_{period}_dist'] = (df['CLOSE'] - df[f'SMA_{period}']) / df['CLOSE'] * 1000

        # EMA
        for period in [12, 26]:
            df[f'EMA_{period}'] = df['CLOSE'].ewm(span=period, adjust=False).mean()
            df[f'EMA_{period}_dist'] = (df['CLOSE'] - df[f'EMA_{period}']) / df['CLOSE'] * 1000

        # RSI  (ИСПРАВЛЕНО: epsilon, иначе loss=0 → rs=inf → inf в модели)
        for period in [7, 14]:
            delta = df['CLOSE'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # ATR
        for period in [14]:
            high_low = df['HIGH'] - df['LOW']
            high_close = np.abs(df['HIGH'] - df['CLOSE'].shift())
            low_close = np.abs(df['LOW'] - df['CLOSE'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
            df[f'ATR_{period}_norm'] = df[f'ATR_{period}'] / df['CLOSE'] * 10000

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_norm'] = df['MACD'] / df['CLOSE'] * 10000

        # Bollinger Bands
        df['BB_middle'] = df['CLOSE'].rolling(window=20).mean()
        df['BB_std'] = df['CLOSE'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 1000
        df['BB_pctB'] = (df['CLOSE'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_position'] = (df['CLOSE'] - df['BB_middle']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)

        # Stochastic
        for period in [14]:
            low_min = df['LOW'].rolling(window=period).min()
            high_max = df['HIGH'].rolling(window=period).max()
            df[f'Stoch_K_{period}'] = (df['CLOSE'] - low_min) / (high_max - low_min + 1e-10) * 100

        # Momentum
        for period in [3, 6, 12]:
            df[f'MOM_{period}'] = (df['CLOSE'] - df['CLOSE'].shift(period)) / df['CLOSE'].shift(period) * 1000

        # Williams %R
        for period in [14]:
            high_max = df['HIGH'].rolling(window=period).max()
            low_min = df['LOW'].rolling(window=period).min()
            df[f'Williams_R_{period}'] = (high_max - df['CLOSE']) / (high_max - low_min + 1e-10) * -100

        # CCI  (ИСПРАВЛЕНО: убрано лишнее /100)
        for period in [20]:
            tp = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'CCI_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # Candle patterns
        df['Body_Ratio'] = np.abs(df['CLOSE'] - df['OPEN']) / (df['HIGH'] - df['LOW'] + 1e-10)
        df['Upper_Wick'] = (df['HIGH'] - df[['OPEN', 'CLOSE']].max(axis=1)) / (df['HIGH'] - df['LOW'] + 1e-10)
        df['Lower_Wick'] = (df[['OPEN', 'CLOSE']].min(axis=1) - df['LOW']) / (df['HIGH'] - df['LOW'] + 1e-10)

        # Time features  (ИСПРАВЛЕНО: добавлен cos, иначе sin неоднозначен)
        df['Hour'] = df['DATETIME'].dt.hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_of_Week'] = df['DATETIME'].dt.dayofweek
        df['DOW_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['DOW_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

        # Volume features
        df['Vol_SMA_20'] = df['TICKVOL'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['TICKVOL'] / (df['Vol_SMA_20'] + 1e-10)

        # Price changes
        df['Last_Bar_Change'] = (df['CLOSE'] - df['OPEN']) / df['OPEN'] * 1000
        df['Gap'] = (df['OPEN'] - df['CLOSE'].shift(1)) / df['CLOSE'].shift(1) * 1000

        # Target
        df['Next_Close'] = df['CLOSE'].shift(-1)

        # ИСПРАВЛЕНО (КРИТИЧНО): экстремумы БУДУЩЕГО бара i+1.
        # Вход на close[i], значит сделка живёт внутри бара i+1 —
        # SL/TP должны проверяться по HIGH/LOW бара i+1, а не своей же свечи i.
        df['Next_High'] = df['HIGH'].shift(-1)
        df['Next_Low'] = df['LOW'].shift(-1)

        df['Target'] = (df['Next_Close'] - df['CLOSE']) / df['CLOSE']
        df['Target_Direction'] = (df['Target'] > 0).astype(int)

        df = df.dropna()

        self.features = df
        print(f"✅ Создано {len(df.columns)} колонок")
        print(f"📊 Осталось {len(df)} записей после удаления NaN")

        return df

    def prepare_data(self):
        """Подготовка данных для обучения"""
        print("\n📈 Подготовка данных...")

        df = self.features.copy()

        # ИСПРАВЛЕНО (КРИТИЧНО): Next_High / Next_Low обязательно в исключениях,
        # иначе будущее утечёт в признаки (lookahead leakage).
        feature_cols = [col for col in df.columns if col not in [
            'DATE', 'TIME', 'DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE',
            'TICKVOL', 'VOL', 'SPREAD',
            'Next_Close', 'Next_High', 'Next_Low',
            'Target', 'Target_Direction'
        ]]

        X = df[feature_cols].values
        y_direction = df['Target_Direction'].values
        y_target = df['Target'].values

        n = len(X)
        split1 = int(n * 0.6)
        split2 = int(n * 0.8)

        X_train = X[:split1]
        X_test1 = X[split1:split2]
        X_test2 = X[split2:]

        y_train_dir = y_direction[:split1]
        y_test1_dir = y_direction[split1:split2]
        y_test2_dir = y_direction[split2:]

        y_train_target = y_target[:split1]
        y_test1_target = y_target[split1:split2]
        y_test2_target = y_target[split2:]

        self.feature_names = feature_cols
        self.splits = {
            'train': (X_train, y_train_dir, y_train_target),
            'test1': (X_test1, y_test1_dir, y_test1_target),
            'test2': (X_test2, y_test2_dir, y_test2_target)
        }

        print(f"✅ Train: {len(X_train)}, Test1: {len(X_test1)}, Test2: {len(X_test2)}")
        print(f"✅ Признаков: {len(feature_cols)}")

        return self.splits

    def train_model(self):
        """Обучение LightGBM модели направления"""
        print("\n🤖 Обучение модели...")

        X_train, y_train_dir, _ = self.splits['train']

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X_train, y_train_dir)

        print("✅ Модель обучена")
        return self.model

    def predict(self, X):
        """Предсказания модели"""
        return self.model.predict_proba(X)[:, 1]

    def backtest(self, dir_prob, data, params):
        """Бэктест стратегии с учетом HIGH/LOW и спреда"""
        conf_threshold = params['conf_threshold']
        sl_pips = params['sl_pips']
        tp_pips = params['tp_pips']
        lot_size = params['lot_size']
        spread_pips = params.get('spread_pips', 1.0)

        pip_value = 0.0001
        sl_dist = sl_pips * pip_value
        tp_dist = tp_pips * pip_value
        spread_cost = spread_pips * pip_value

        equity = 10000
        peak = equity
        max_dd = 0
        equity_curve = []
        trades = []
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0

        for i in range(len(dir_prob)):
            confidence = abs(dir_prob[i] - 0.5) * 2

            if confidence < conf_threshold:
                equity_curve.append(equity)
                continue

            is_long = dir_prob[i] > 0.5
            entry = data.iloc[i]['CLOSE']

            # ИСПРАВЛЕНО (КРИТИЧНО): экстремумы бара i+1, а не бара i.
            # Раньше брались HIGH[i]/LOW[i] — свеча, которая закрылась ДО входа.
            high = data.iloc[i]['Next_High']
            low = data.iloc[i]['Next_Low']
            next_close = data.iloc[i]['Next_Close']

            if is_long:
                # Сначала стоп, потом тейк (консервативно)
                if low <= entry - sl_dist:
                    exit_price = entry - sl_dist
                elif high >= entry + tp_dist:
                    exit_price = entry + tp_dist
                else:
                    exit_price = next_close
                gross_pnl = (exit_price - entry) * lot_size
            else:
                if high >= entry + sl_dist:
                    exit_price = entry + sl_dist
                elif low <= entry - tp_dist:
                    exit_price = entry - tp_dist
                else:
                    exit_price = next_close
                gross_pnl = (entry - exit_price) * lot_size

            trade_pnl = gross_pnl - spread_cost * lot_size
            equity += trade_pnl

            if trade_pnl > 1e-9:
                wins += 1
                total_profit += trade_pnl
            elif trade_pnl < -1e-9:
                losses += 1
                total_loss += abs(trade_pnl)

            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            trades.append({
                'datetime': data.iloc[i]['DATETIME'],
                'side': 'BUY' if is_long else 'SELL',
                'entry': entry,
                'exit': exit_price,
                'pnl': trade_pnl,
                'confidence': confidence,
                'equity': equity
            })

            equity_curve.append(equity)

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)

        # Sharpe по сделкам (приближённый)
        sharpe = 0
        if len(trades) > 1:
            returns = [t['pnl'] / t['equity'] for t in trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                first_dt = trades[0]['datetime']
                last_dt = trades[-1]['datetime']
                years = (last_dt - first_dt).total_seconds() / (365.25 * 24 * 3600)
                trades_per_year = len(trades) / years if years > 0 else 0
                sharpe = (avg_return / std_return) * np.sqrt(trades_per_year)

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': equity,
            'total_return': ((equity - 10000) / 10000) * 100,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'max_dd': max_dd * 100,
            'profit_factor': profit_factor,
            'sharpe': sharpe
        }

    def grid_search(self, X_test1, data_test1):
        """Grid search параметров на Test1"""
        print("\n🔍 Grid search на Test1...")

        dir_prob = self.predict(X_test1)

        conf_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        sl_pips_list = [10, 15, 20, 25, 30]
        tp_pips_list = [15, 20, 30, 40, 50]
        spread_pips_list = [0.5, 1.0, 1.5]

        results = []
        total = len(conf_thresholds) * len(sl_pips_list) * len(tp_pips_list) * len(spread_pips_list)
        count = 0

        for conf in conf_thresholds:
            for sl in sl_pips_list:
                for tp in tp_pips_list:
                    if tp <= sl:
                        continue
                    for spread in spread_pips_list:
                        params = {
                            'conf_threshold': conf,
                            'sl_pips': sl,
                            'tp_pips': tp,
                            'lot_size': 100000,
                            'spread_pips': spread
                        }

                        result = self.backtest(dir_prob, data_test1, params)

                        if result['total_trades'] >= 50 and result['win_rate'] >= 48:
                            score = (result['sharpe'] * 0.4 +
                                     result['profit_factor'] * 0.3 +
                                     result['win_rate'] * 0.2 -
                                     result['max_dd'] * 0.1)

                            results.append({'params': params, **result, 'score': score})

                        count += 1
                        if count % 50 == 0:
                            print(f"  Прогресс: {count}/{total}")

        if not results:
            print("⚠️ Не найдено стратегий, удовлетворяющих фильтрам. Смягчи условия.")
            return []

        results.sort(key=lambda x: x['score'], reverse=True)
        top_10 = results[:10]

        print(f"✅ Найдено {len(results)} стратегий, удовлетворяющих фильтрам")
        print(f"🏆 Топ-10 выбрано")
        return top_10

    def run_top_10_strategies(self):
        """Запуск топ-10 стратегий на Test2"""
        print("\n🚀 Запуск топ-10 стратегий на Test2...")

        n = len(self.features)
        split1 = int(n * 0.6)
        split2 = int(n * 0.8)

        X_test1, _, _ = self.splits['test1']
        data_test1 = self.features.iloc[split1:split2].reset_index(drop=True)

        X_test2, _, _ = self.splits['test2']
        data_test2 = self.features.iloc[split2:].reset_index(drop=True)

        top_10 = self.grid_search(X_test1, data_test1)

        if not top_10:
            print("❌ Топ-10 пуст, пропускаем запуск на Test2")
            return []

        predictions_test2 = self.predict(X_test2)

        results = []
        for i, strategy in enumerate(top_10):
            result = self.backtest(predictions_test2, data_test2, strategy['params'])
            result['strategy_id'] = i + 1
            result['params'] = strategy['params']
            results.append(result)
            print(f"  Стратегия #{i+1}: Return ${result['final_equity']-10000:.0f}, "
                  f"WR {result['win_rate']:.1f}%, Trades {result['total_trades']}")

        return results

    def plot_results(self, results):
        """Визуализация результатов"""
        print("\n📊 Построение графиков...")

        fig = plt.figure(figsize=(20, 12))

        # Equity curves всех стратегий
        ax1 = plt.subplot(2, 2, 1)
        for result in results:
            ax1.plot(result['equity_curve'], label=f"#{result['strategy_id']}",
                     alpha=0.7, linewidth=1.5)
        ax1.set_title('All Strategies Equity (Test 2)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bar #')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Средняя эквити (ИСПРАВЛЕНО: это НЕ независимые стратегии —
        # все 10 наборов параметров гоняются на одном и том же dir_prob)
        ax2 = plt.subplot(2, 2, 2)
        max_len = max(len(r['equity_curve']) for r in results)
        ensemble = np.zeros(max_len)
        for result in results:
            eq = result['equity_curve']
            ensemble[:len(eq)] += eq
        ensemble /= len(results)
        ax2.plot(ensemble, color='#10b981', linewidth=2,
                 label='Average across params (correlated)')
        ax2.set_title('Average Equity (Params Ensemble)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bar #')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Win rate comparison
        ax3 = plt.subplot(2, 2, 3)
        strategy_ids = [r['strategy_id'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        colors = ['#4ade80' if wr > 50 else '#f87171' for wr in win_rates]
        ax3.bar(strategy_ids, win_rates, color=colors, alpha=0.7)
        ax3.set_title('Win Rate by Strategy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Strategy #')
        ax3.set_ylabel('Win Rate (%)')
        ax3.axhline(y=50, color='yellow', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')

        # Returns comparison
        ax4 = plt.subplot(2, 2, 4)
        returns = [r['total_return'] for r in results]
        colors = ['#4ade80' if r > 0 else '#f87171' for r in returns]
        ax4.bar(strategy_ids, returns, color=colors, alpha=0.7)
        ax4.set_title('Total Return by Strategy (%)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Strategy #')
        ax4.set_ylabel('Return (%)')
        ax4.axhline(y=0, color='yellow', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('trading_results.png', dpi=150, bbox_inches='tight')
        print("✅ График сохранен: trading_results.png")

        # Детальные эквити
        fig2, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        for i, result in enumerate(results[:10]):
            axes[i].plot(result['equity_curve'], color='#3b82f6', linewidth=1.5)
            axes[i].set_title(
                f"Strategy #{result['strategy_id']}\n"
                f"${result['final_equity']-10000:.0f} ({result['win_rate']:.1f}%)",
                fontsize=10)
            axes[i].set_xlabel('Bar #')
            axes[i].set_ylabel('Equity ($)')
            axes[i].grid(True, alpha=0.3)

        # Скрыть пустые subplot, если стратегий меньше 10
        for j in range(len(results), 10):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig('strategy_details.png', dpi=150, bbox_inches='tight')
        print("✅ Детальные графики сохранены: strategy_details.png")

    def print_summary(self, results):
        """Печать сводки результатов"""
        print("\n" + "="*80)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("="*80)

        print(f"\nВсего стратегий: {len(results)}")

        best = max(results, key=lambda x: x['final_equity'])
        print(f"🏆 Лучшая стратегия: #{best['strategy_id']}")
        print(f"   Return: ${best['final_equity']-10000:.0f}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Trades: {best['total_trades']}")
        print(f"   Max DD: {best['max_dd']:.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   Params: {best['params']}")

        avg_wr = np.mean([r['win_rate'] for r in results])
        print(f"\n📊 Средняя Win Rate: {avg_wr:.1f}%")

        avg_return = np.mean([r['final_equity'] for r in results]) - 10000
        print(f"💰 Средняя Return по наборам параметров: ${avg_return:.0f}")

        print("\n" + "="*80)
        print("ДЕТАЛЬНАЯ ТАБЛИЦА")
        print("="*80)
        print(f"{'#':<4} {'Return':<12} {'WinRate':<10} {'Trades':<8} {'MaxDD':<10} {'Sharpe':<8} {'PF':<8}")
        print("-"*80)
        for r in results:
            ret_str = f"${r['final_equity']-10000:.0f}"
            print(f"{r['strategy_id']:<4} {ret_str:<12} {r['win_rate']:<10.1f} "
                  f"{r['total_trades']:<8} {r['max_dd']:<10.1f} {r['sharpe']:<8.2f} "
                  f"{r['profit_factor']:<8.2f}")
        print("="*80)


def main():
    print("\n" + "="*80)
    print("EURUSD H1 ML TRADING SYSTEM")
    print("="*80)

    csv_path = 'EURUSD_H1_202401020000_202608262100.csv'

    system = EURUSDMLSystem(csv_path)
    system.load_data()
    system.calculate_features()
    system.prepare_data()
    system.train_model()

    results = system.run_top_10_strategies()

    if results:
        system.plot_results(results)
        system.print_summary(results)

    print("\n✅ Завершено!")


if __name__ == '__main__':
    main()