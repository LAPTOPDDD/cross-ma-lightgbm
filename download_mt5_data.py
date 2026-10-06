import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time

# Попытка импорта MetaTrader5
try:
    import MetaTrader5 as mt5
except ImportError:
    print("Пакет MetaTrader5 не установлен. Устанавливаю...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
    import MetaTrader5 as mt5

# =============================================================================
# 1. ПОДКЛЮЧЕНИЕ К MT5
# =============================================================================
print("Подключение к MetaTrader 5...")
if not mt5.initialize():
    print("ОШИБКА: Не удалось подключиться к MetaTrader 5")
    print("Убедитесь, что MT5 запущен и авторизован.")
    print(f"Код ошибки: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print(f"Успешно подключено к MT5")
term_info = mt5.terminal_info()
acc_info = mt5.account_info()
if term_info:
    print(f"Терминал: {term_info.name if hasattr(term_info, 'name') else 'N/A'}")
if acc_info:
    print(f"Логин: {acc_info.login}")
    print(f"Баланс: {acc_info.balance}")

# =============================================================================
# 2. НАСТРОЙКИ
# =============================================================================
SYMBOLS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'],
    'commodities': ['XAUUSD', 'XAGUSD'],
    'crypto': ['BTCUSD', 'ETHUSD']
}

TIMEFRAMES = {
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}

# Период данных: с 2024-01-01 по сегодня
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime.now()

OUTPUT_DIR = "mt5_downloaded_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 3. ФУНКЦИЯ ВЫГРУЗКИ
# =============================================================================
def download_symbol_data(symbol, tf_name, tf_code):
    """Скачивает бары из MT5 и сохраняет в CSV."""
    print(f"\n  Скачиваю {symbol} {tf_name}...")

    # Проверяем, доступен ли символ
    if not mt5.symbol_select(symbol, True):
        print(f"  ПРЕДУПРЕЖДЕНИЕ: Не удалось выбрать {symbol}, возможно, он недоступен")
        return False

    # Скачиваем бары
    rates = mt5.copy_rates_range(symbol, tf_code, START_DATE, END_DATE)

    if rates is None or len(rates) == 0:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: Нет данных для {symbol} {tf_name}")
        return False

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Переименуем колонки под формат MT4
    df = df[['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Real_Volume']

    # Сохраняем
    filename = f"{OUTPUT_DIR}/{symbol}_{tf_name}_{START_DATE.strftime('%Y%m%d')}_{END_DATE.strftime('%Y%m%d')}.csv"
    df.to_csv(filename)
    print(f"  Сохранено: {filename} ({len(df)} баров)")
    return True

# =============================================================================
# 4. ОСНОВНОЙ ЦИКЛ
# =============================================================================
print(f"\nНачинаем выгрузку данных с {START_DATE.date()} по {END_DATE.date()}")
print(f"Символы: {sum(SYMBOLS.values(), [])}")
print(f"Таймфреймы: {list(TIMEFRAMES.keys())}")

total_files = 0
failed = []

for category, symbols in SYMBOLS.items():
    print(f"\n{'='*60}")
    print(f"Категория: {category}")
    print(f"{'='*60}")
    for symbol in symbols:
        for tf_name, tf_code in TIMEFRAMES.items():
            success = download_symbol_data(symbol, tf_name, tf_code)
            if success:
                total_files += 1
            else:
                failed.append(f"{symbol}_{tf_name}")
            time.sleep(0.5)  # Небольшая пауза, чтобы не перегружать MT5

# =============================================================================
# 5. ИТОГ
# =============================================================================
mt5.shutdown()

print(f"\n{'='*60}")
print(f"ВЫГРУЗКА ЗАВЕРШЕНА")
print(f"{'='*60}")
print(f"Успешно сохранено файлов: {total_files}")
print(f"Папка: {os.path.abspath(OUTPUT_DIR)}")

if failed:
    print(f"\nНе удалось скачать ({len(failed)}):")
    for f in failed:
        print(f"  - {f}")
else:
    print("\nВсе данные успешно скачаны!")
