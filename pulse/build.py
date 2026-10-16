"""Обновление данных, KPI, графиков, README и данных для веб-страницы.

    python -m pulse.build            # скачать новое с MOEX и всё пересобрать
    python -m pulse.build --offline  # только пересобрать из data/*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .iss import SERIES, Series, fetch

ROOT = Path(__file__).resolve().parents[1]
DATA, CHARTS, DOCS = ROOT / "data", ROOT / "charts", ROOT / "docs"
README = ROOT / "README.md"
FIRST_DATE = "2024-01-01"

INK, MUTED, GRID = "#1d1d1f", "#86868b", "#e8e8ed"
PALETTE = ["#0071e3", "#bf4800", "#248a3d", "#8e44ad", "#a1a1a6"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 160, "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
})


# ---------- данные ----------
def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Явные типы: пустая таблица в pandas — это object, и после concat цены стали бы строками."""
    return pd.DataFrame({"date": pd.to_datetime(df["date"]),
                         "close": pd.to_numeric(df["close"], errors="coerce").astype(float)})


def load_local(s: Series) -> pd.DataFrame:
    f = DATA / f"{s.key}.csv"
    if f.exists():
        return _typed(pd.read_csv(f))
    return _typed(pd.DataFrame({"date": [], "close": []}))


def update(s: Series, get=None) -> tuple[pd.DataFrame, str | None]:
    """Докачать новые точки. При ошибке сети оставляем то, что было, и возвращаем текст ошибки."""
    old = load_local(s)
    since = FIRST_DATE if old.empty else (old.date.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        new = fetch(s, since) if get is None else fetch(s, since, get=get)
    except Exception as e:  # noqa: BLE001
        return old, f"{s.key}: {type(e).__name__}: {e}"
    parts = [x for x in (old, _typed(new)) if not x.empty]
    if not parts:
        return old, f"{s.key}: биржа вернула пустую историю"
    df = pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df = _typed(df.dropna())
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / f"{s.key}.csv", index=False, date_format="%Y-%m-%d")
    return df, None


# ---------- сплиты ----------
NICE_FACTORS = [2, 3, 4, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def adjust_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Корректирует цены акций на сплиты и консолидации.

    Изменение цены больше чем в 3 раза за день на российском рынке невозможно из-за ценовых
    ограничений биржи — такой скачок означает сплит. Коэффициент округляем до ближайшего
    «круглого» (10:1, 1:5000 …) и пересчитываем все цены до даты события.
    """
    df = df.sort_values("date").reset_index(drop=True)
    close = df.close.to_numpy(dtype=float).copy()
    events = []
    for i in range(1, len(close)):
        ratio = close[i] / close[i - 1]
        if 1 / 3 < ratio < 3:
            continue
        up = ratio >= 3
        raw = ratio if up else 1 / ratio
        factor = min(NICE_FACTORS, key=lambda f: abs(np.log(f / raw)))
        mult = factor if up else 1 / factor
        close[:i] *= mult
        events.append(dict(date=df.date[i].strftime("%d.%m.%Y"),
                           kind="консолидация" if up else "сплит",
                           ratio=f"{factor}:1" if not up else f"1:{factor}"))
    return df.assign(close=close), events


# ---------- KPI ----------
def value_at(df: pd.DataFrame, when: pd.Timestamp) -> float | None:
    """Последнее закрытие не позже даты when."""
    prev = df[df.date <= when]
    return float(prev.close.iloc[-1]) if len(prev) else None


def kpi(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    def chg(base):
        return None if base in (None, 0) else float(last.close / base - 1)
    year_start = pd.Timestamp(year=last.date.year, month=1, day=1) - pd.Timedelta(days=1)
    return dict(date=last.date.strftime("%d.%m.%Y"), last=float(last.close),
                d1=chg(float(df.close.iloc[-2]) if len(df) > 1 else None),
                m1=chg(value_at(df, last.date - pd.Timedelta(days=30))),
                ytd=chg(value_at(df, year_start)))


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1%}".replace(".", ",").replace("-", "−")


def fmt_val(x: float, unit: str) -> str:
    s = f"{x:,.2f}" if x < 1000 else f"{x:,.0f}"
    return s.replace(",", " ").replace(".", ",") + (f" {unit}" if unit and unit != "п." else "")


# ---------- графики ----------
def _finish(ax, fig, name: str) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    ax.tick_params(length=0)
    fig.tight_layout()
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, facecolor="white")
    plt.close(fig)


def chart_line(df: pd.DataFrame, title: str, name: str, color: str = PALETTE[0]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(df.date, df.close, color=color, lw=1.6)
    ax.fill_between(df.date, df.close, df.close.min() * 0.98, color=color, alpha=0.06, lw=0)
    ax.set_title(title)
    ax.set_ylim(bottom=df.close.min() * 0.98)
    _finish(ax, fig, name)


def chart_rebased(frames: dict[str, pd.DataFrame], title: str, name: str) -> None:
    """Несколько бумаг на одной шкале: 100 = закрытие на первую дату."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for (label, df), color in zip(frames.items(), PALETTE):
        if df.empty:
            continue
        ax.plot(df.date, df.close / df.close.iloc[0] * 100, color=color, lw=1.5, label=label)
    ax.axhline(100, color=MUTED, lw=0.6, ls=":")
    ax.set_title(title)
    ax.legend(ncol=len(frames), loc="upper left", fontsize=8)
    _finish(ax, fig, name)


# ---------- README и веб ----------
def render_table(rows: list[tuple[Series, dict]]) -> str:
    lines = ["| | Значение | За день | За месяц | С начала года |", "|---|---:|---:|---:|---:|"]
    for s, k in rows:
        lines.append(f"| {s.title} | {fmt_val(k['last'], s.unit)} | {fmt_pct(k['d1'])} | {fmt_pct(k['m1'])} | {fmt_pct(k['ytd'])} |")
    return "\n".join(lines)


def replace_block(text: str, body: str, start="<!-- PULSE:START -->", end="<!-- PULSE:END -->") -> str:
    if start not in text or end not in text:
        raise ValueError("В README нет маркеров PULSE:START / PULSE:END")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def main(offline: bool = False) -> int:
    frames, errors = {}, []
    for s in SERIES:
        if offline:
            frames[s.key] = load_local(s)
        else:
            frames[s.key], err = update(s)
            if err:
                errors.append(err)
                print("WARN", err, file=sys.stderr)
    if not offline and len(errors) == len(SERIES):
        print("Биржа не ответила ни по одному ряду — ничего не обновляем", file=sys.stderr)
        return 1
    frames = {k: v for k, v in frames.items() if len(v) > 1}
    corporate = {}
    for s in SERIES:
        if s.group == "Банки" and s.key in frames:
            frames[s.key], ev = adjust_splits(frames[s.key])
            if ev:
                corporate[s.title] = ev
    if not frames:
        print("Нет ни одного ряда данных", file=sys.stderr)
        return 1

    rows = [(s, kpi(frames[s.key])) for s in SERIES if s.key in frames]
    since = pd.Timestamp.now().normalize() - pd.DateOffset(months=18)
    recent = {k: v[v.date >= since] for k, v in frames.items()}
    if "IMOEX" in recent:
        chart_line(recent["IMOEX"], "Индекс МосБиржи", "imoex.png")
    if "RGBI" in recent:
        chart_line(recent["RGBI"], "Индекс гособлигаций RGBI — растёт, когда рынок ждёт снижения ставки",
                   "rgbi.png", PALETTE[2])
    if "CNYRUB" in recent:
        chart_line(recent["CNYRUB"], "Юань к рублю (CNYRUB_TOM)", "cnyrub.png", PALETTE[1])
    year = pd.Timestamp(year=pd.Timestamp.now().year, month=1, day=1)
    banks = {s.title: frames[s.key][frames[s.key].date >= year] for s in SERIES if s.group == "Банки" and s.key in frames}
    if banks:
        chart_rebased(banks, f"Акции финансового сектора с начала {year.year} года (100 = 1 января)", "banks.png")

    # в README и на странице — дата последних торгов, а не время сборки: если новых торгов нет,
    # файлы не меняются и робот ничего не коммитит
    last_trade = max(frames[s.key].date.max() for s, _ in rows).strftime("%d.%m.%Y")
    body = "\n".join([
        f"*Последние торги: {last_trade}. Обновляется автоматически каждую ночь.*", "",
        render_table(rows), "",
        *([("Цены акций скорректированы на корпоративные события: "
            + "; ".join(f"{t} — {e['kind']} {e['ratio']} ({e['date']})" for t, evs in corporate.items() for e in evs) + "."), ""]
          if corporate else []),
        '<img src="charts/imoex.png" width="100%">', "",
        '<img src="charts/rgbi.png" width="100%">', "",
        '<img src="charts/banks.png" width="100%">', "",
        '<img src="charts/cnyrub.png" width="100%">',
    ])
    README.write_text(replace_block(README.read_text(encoding="utf-8"), body), encoding="utf-8")

    DOCS.mkdir(exist_ok=True)
    web = dict(updated=last_trade, kpi=[dict(key=s.key, title=s.title, group=s.group, unit=s.unit, **k) for s, k in rows],
               series={k: dict(dates=v.date.dt.strftime("%Y-%m-%d").tolist(), close=v.close.round(4).tolist())
                       for k, v in recent.items()})
    (DOCS / "data.json").write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(rows)} рядов, последние торги {last_trade}" + (f", ошибок: {len(errors)}" if errors else ""))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    sys.exit(main(p.parse_args().offline))
