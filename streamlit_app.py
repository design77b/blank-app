import io
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Анализ лотереи 6 из 37", page_icon="🎲", layout="centered")

st.title("🎲 Анализ лотереи «6 из 37»")
st.caption(
    "⚠️ Важно: лотерея случайна. Приложение не предсказывает выигрыш и не увеличивает вероятность, "
    "а помогает выбирать осмысленно на основе статистики прошлых тиражей."
)

st.divider()

# ---------- Helpers ----------

def _try_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV reader for common encodings and separators.
    """
    raw = uploaded_file.getvalue()
    # Try encodings commonly used for Russian locale exports
    encodings = ["utf-8-sig", "utf-8", "cp1251", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            # auto-separator using python engine
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Не удалось прочитать CSV (кодировка/разделитель). Ошибка: {last_err}")


def _detect_columns(df: pd.DataFrame):
    """
    Detect 6 main number columns and optional bonus column.
    User described: main numbers are in columns C..H, bonus in I.
    But we also support other structures.

    Returns: (main_cols, bonus_col_or_None)
    """
    # Prefer exact numeric column names like '1'..'6'
    cols = list(df.columns)

    # Candidate main cols: exact '1'..'6'
    if all(str(i) in cols for i in range(1, 7)):
        main_cols = [str(i) for i in range(1, 7)]
    else:
        # Otherwise: take columns by position C..H (0-based: 2..7), if exists
        if len(cols) >= 8:
            main_cols = cols[2:8]
        else:
            # Fallback: choose first 6 mostly-numeric columns
            numeric_score = []
            for c in cols:
                s = pd.to_numeric(df[c], errors="coerce")
                numeric_score.append((c, s.notna().mean()))
            numeric_score.sort(key=lambda x: x[1], reverse=True)
            main_cols = [c for c, _ in numeric_score[:6]]

    # Bonus: prefer a col that looks like bonus (1..7 or 1..8)
    bonus_col = None

    # If there is a 9th column (I position), prefer it
    if len(cols) >= 9:
        candidate = cols[8]
        cand_vals = pd.to_numeric(df[candidate], errors="coerce")
        if cand_vals.notna().mean() > 0.8:
            bonus_col = candidate

    # Otherwise: scan columns for value-range 1..7/8
    if bonus_col is None:
        for c in cols:
            if c in main_cols:
                continue
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            mn, mx = int(s.min()), int(s.max())
            # many lotteries use 1..7; sometimes 1..8 appears (your data shows 8 occasionally)
            if mn >= 1 and mx <= 8 and s.mean() <= 6.5:
                bonus_col = c
                break

    return main_cols, bonus_col


def _clean_numbers(df: pd.DataFrame, main_cols, bonus_col):
    """
    Convert to integers, validate ranges.
    """
    d = df.copy()

    for c in main_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Drop rows where any of the 6 numbers is missing
    d = d.dropna(subset=main_cols)

    # Convert to int
    d[main_cols] = d[main_cols].astype(int)

    # Validate range 1..37
    ok = True
    bad_rows = ((d[main_cols] < 1) | (d[main_cols] > 37)).any(axis=1)
    if bad_rows.any():
        ok = False
        d = d.loc[~bad_rows].copy()

    # Bonus
    if bonus_col is not None:
        d[bonus_col] = pd.to_numeric(d[bonus_col], errors="coerce")
        d = d.dropna(subset=[bonus_col])
        d[bonus_col] = d[bonus_col].astype(int)
        # allow 1..8 (your file shows 8 sometimes)
        badb = (d[bonus_col] < 1) | (d[bonus_col] > 8)
        d = d.loc[~badb].copy()

    return d, ok


def _freq_table(d: pd.DataFrame, main_cols):
    all_nums = d[main_cols].to_numpy().flatten()
    freq = Counter(all_nums)
    freq_df = pd.DataFrame({"Число": list(freq.keys()), "Выпадений": list(freq.values())})
    freq_df = freq_df.sort_values("Выпадений", ascending=False).reset_index(drop=True)
    return freq_df


def _bonus_freq_table(d: pd.DataFrame, bonus_col):
    if bonus_col is None:
        return None
    freq = Counter(d[bonus_col].to_numpy().tolist())
    bdf = pd.DataFrame({"Бонус": list(freq.keys()), "Выпадений": list(freq.values())})
    bdf = bdf.sort_values("Выпадений", ascending=False).reset_index(drop=True)
    return bdf


def _avoid_patterns(nums):
    nums = sorted(nums)
    # avoid 3+ consecutive numbers
    consec = 0
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            consec += 1
    if consec >= 2:
        return False
    # avoid too many from 1..12 (date-like)
    if sum(1 for x in nums if x <= 12) >= 5:
        return False
    # avoid all even/odd
    ev = sum(1 for x in nums if x % 2 == 0)
    if ev == 0 or ev == 6:
        return False
    return True


def _gen_weighted(numbers, weights, k=6, tries=5000):
    """
    Weighted sampling without replacement with simple filters.
    """
    numbers = np.array(numbers)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    for _ in range(tries):
        pick = np.random.choice(numbers, size=k, replace=False, p=weights)
        pick = sorted(map(int, pick))
        if _avoid_patterns(pick):
            return pick
    # fallback
    pick = sorted(map(int, np.random.choice(numbers, size=k, replace=False)))
    return pick


def _strategy_frequent(freq_df):
    # Use top 20 as pool, weighted by frequency
    pool = freq_df.head(20)
    return _gen_weighted(pool["Число"].tolist(), pool["Выпадений"].tolist(), k=6)


def _strategy_balanced(freq_df):
    # Weighted by freq but enforce spread across ranges
    pool = freq_df.head(28)
    nums = pool["Число"].tolist()
    w = pool["Выпадений"].tolist()

    # create buckets
    buckets = {
        "low": [n for n in nums if 1 <= n <= 12],
        "mid": [n for n in nums if 13 <= n <= 24],
        "high": [n for n in nums if 25 <= n <= 37],
    }

    # pick 2 from each bucket using weights
    def pick_from(bucket, m):
        if len(bucket) < m:
            return random.sample(nums, m)
        bw = [pool.loc[pool["Число"] == n, "Выпадений"].iloc[0] for n in bucket]
        return _gen_weighted(bucket, bw, k=m, tries=2000)

    chosen = []
    chosen += pick_from(buckets["low"], 2)
    chosen += pick_from(buckets["mid"], 2)
    chosen += pick_from(buckets["high"], 2)
    chosen = sorted(set(chosen))
    # if duplicates reduced count, fill from pool
    while len(chosen) < 6:
        extra = _gen_weighted(nums, w, k=1, tries=500)[0]
        chosen = sorted(set(chosen + [extra]))
    return chosen[:6]


def _strategy_cold(freq_df):
    # Cold = bottom 20 by frequency (but exclude extremely rare outliers by taking bottom 26 and sampling)
    pool = freq_df.sort_values("Выпадений", ascending=True).head(26)
    # Slight preference to "not the absolute coldest" by adding 1 to weights
    # so it doesn't overfocus on extreme tails
    weights = (pool["Выпадений"] + 1).tolist()
    return _gen_weighted(pool["Число"].tolist(), weights, k=6)


def _strategy_anti_popular(freq_df):
    # Avoid date-like + round numbers + too popular: use mid-frequency region
    mid = freq_df.iloc[10:30].copy()
    mid_nums = mid["Число"].tolist()
    mid_w = mid["Выпадений"].tolist()

    def ok_extra(pick):
        # avoid round numbers concentration
        if sum(1 for x in pick if x % 10 == 0) >= 2:
            return False
        # avoid too many <= 12
        if sum(1 for x in pick if x <= 12) >= 4:
            return False
        return True

    for _ in range(5000):
        pick = _gen_weighted(mid_nums, mid_w, k=6, tries=100)
        if ok_extra(pick):
            return pick
    return _gen_weighted(mid_nums, mid_w, k=6)


def _strategy_random():
    nums = sorted(random.sample(range(1, 38), 6))
    # try a couple times to avoid obvious patterns
    for _ in range(200):
        if _avoid_patterns(nums):
            break
        nums = sorted(random.sample(range(1, 38), 6))
    return nums


def _pick_bonus(bonus_df):
    if bonus_df is None or bonus_df.empty:
        return None, []
    top = bonus_df.head(3)["Бонус"].tolist()
    primary = int(top[0])
    alternatives = [int(x) for x in top[1:]]
    return primary, alternatives


# ---------- UI ----------

uploaded = st.file_uploader("📂 Загрузите CSV-файл с историей тиражей", type=["csv"])

if not uploaded:
    st.info("Загрузите CSV, чтобы увидеть статистику и сгенерировать комбинации.")
    st.stop()

try:
    df_raw = _try_read_csv(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

main_cols, bonus_col = _detect_columns(df_raw)

with st.expander("ℹ️ Какой формат распознан", expanded=False):
    st.write("**Колонки с 6 основными числами:**", main_cols)
    st.write("**Колонка бонусного числа:**", bonus_col if bonus_col is not None else "не найдена (необязательно)")

df, range_ok = _clean_numbers(df_raw, main_cols, bonus_col)

if df.empty:
    st.error("После проверки данных не осталось корректных строк. Проверьте CSV.")
    st.stop()

if not range_ok:
    st.warning("В некоторых строках были значения вне диапазона 1..37 — такие строки пропущены.")

st.success(f"✅ Файл загружен. Строк (игр) в базе: {len(df):,}".replace(",", " "))

# Stats
freq_df = _freq_table(df, main_cols)
bonus_df = _bonus_freq_table(df, bonus_col)

st.subheader("📊 Краткая статистика")
top10 = freq_df.head(10)["Число"].tolist()
cold10 = freq_df.sort_values("Выпадений", ascending=True).head(10)["Число"].tolist()

st.write("**Самые частые числа (топ-10):**", ", ".join(map(str, top10)))
st.write("**Самые редкие числа (топ-10):**", ", ".join(map(str, cold10)))

bonus_primary, bonus_alts = _pick_bonus(bonus_df)
if bonus_primary is not None:
    st.write(f"**Самое частое бонусное число:** {bonus_primary}")
    if bonus_alts:
        st.write("**Альтернативы:**", ", ".join(map(str, bonus_alts)))
else:
    st.write("**Бонусное число:** колонка не распознана (можно игнорировать).")

with st.expander("📋 Таблица частот (все числа)", expanded=False):
    st.dataframe(freq_df, use_container_width=True)

if bonus_df is not None:
    with st.expander("📋 Частоты бонусного числа", expanded=False):
        st.dataframe(bonus_df, use_container_width=True)

st.divider()
st.subheader("🎯 Генерация комбинаций")

col1, col2 = st.columns(2)
with col1:
    n_sets = st.slider("Сколько комбинаций показать?", 1, 10, 5)
with col2:
    use_bonus = st.checkbox("Показывать бонусное число", value=(bonus_primary is not None))

strategy = st.radio(
    "Выберите стратегию:",
    [
        "🎯 Частотная",
        "⚖️ Сбалансированная",
        "❄️ Холодные числа",
        "🚫 Минимум совпадений с другими игроками",
        "🎲 Случайная (контроль)",
    ],
    index=1,
)

def generate_one():
    if strategy.startswith("🎯"):
        nums = _strategy_frequent(freq_df)
        why = "Выбраны числа с высокой частотой выпадений (статистический подход)."
    elif strategy.startswith("⚖️"):
        nums = _strategy_balanced(freq_df)
        why = "Сбалансировано по диапазонам и чётности, избегает очевидных шаблонов."
    elif strategy.startswith("❄️"):
        nums = _strategy_cold(freq_df)
        why = "Упор на числа с более низкой исторической частотой (альтернативная стратегия)."
    elif strategy.startswith("🚫"):
        nums = _strategy_anti_popular(freq_df)
        why = "Комбинация менее «популярная» у игроков (даты/круглые/красивые шаблоны исключаются)."
    else:
        nums = _strategy_random()
        why = "Полностью случайный выбор (для сравнения)."

    bonus = None
    if use_bonus and bonus_primary is not None:
        # rotate between top-3 a bit to avoid always same
        options = [bonus_primary] + bonus_alts
        bonus = random.choice(options) if options else bonus_primary
    return nums, bonus, why

if st.button("Сгенерировать"):
    results = []
    seen = set()
    attempts = 0
    while len(results) < n_sets and attempts < 200:
        attempts += 1
        nums, bonus, why = generate_one()
        key = (tuple(nums), bonus)
        if key in seen:
            continue
        seen.add(key)
        results.append((nums, bonus, why))

    for i, (nums, bonus, why) in enumerate(results, start=1):
        st.markdown(f"### Комбинация {i}")
        st.write("**Основные числа:**", " – ".join(map(str, nums)))
        if use_bonus and bonus is not None:
            st.write("**Бонус:**", bonus)
        st.caption("Почему: " + why)

st.divider()
st.caption("Если появится ошибка чтения CSV — пришлите файл/скрин, я подстрою распознавание под Ваш формат.")
