"""Мини-клиент MOEX ISS (https://iss.moex.com) — открытого API Московской биржи.

Используются только исторические ряды закрытия: /iss/history/engines/{engine}/markets/{market}/...
API отдаёт данные страницами по 100 строк, курсор лежит в блоке history.cursor.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import pandas as pd

BASE = "https://iss.moex.com/iss"


@dataclass(frozen=True)
class Series:
    key: str          # имя ряда в данных
    title: str        # подпись на графиках
    path: str         # путь до бумаги в ISS
    group: str        # блок дашборда
    unit: str = ""


SERIES = [
    Series("IMOEX", "Индекс МосБиржи", "engines/stock/markets/index/securities/IMOEX", "Рынок", "п."),
    Series("RGBI", "Индекс гособлигаций RGBI", "engines/stock/markets/index/securities/RGBI", "Рынок", "п."),
    Series("CNYRUB", "Юань", "engines/currency/markets/selt/boards/CETS/securities/CNYRUB_TOM", "Валюта", "₽"),
    Series("SBER", "Сбербанк", "engines/stock/markets/shares/boards/TQBR/securities/SBER", "Банки", "₽"),
    Series("T", "Т-Технологии", "engines/stock/markets/shares/boards/TQBR/securities/T", "Банки", "₽"),
    Series("VTBR", "ВТБ", "engines/stock/markets/shares/boards/TQBR/securities/VTBR", "Банки", "₽"),
    Series("MOEX", "Мосбиржа", "engines/stock/markets/shares/boards/TQBR/securities/MOEX", "Банки", "₽"),
]


def _get(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ru-market-pulse/1.0 (+github.com/antonshkelll)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — сеть ненадёжна, пробуем ещё раз
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def parse_history(payload: dict) -> pd.DataFrame:
    """Блок history → DataFrame(date, close). Пустые CLOSE (неторговые дни индексов) выкидываем."""
    block = payload["history"]
    df = pd.DataFrame(block["data"], columns=block["columns"])
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({"date": pd.to_datetime(df["TRADEDATE"]), "close": pd.to_numeric(df["CLOSE"], errors="coerce")})
    return out.dropna().drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch(series: Series, since: str, get=_get) -> pd.DataFrame:
    """Вся история ряда начиная с даты since (YYYY-MM-DD), с учётом пагинации."""
    frames, start = [], 0
    while True:
        q = urllib.parse.urlencode({"from": since, "start": start, "iss.meta": "off",
                                    "history.columns": "TRADEDATE,CLOSE"})
        payload = get(f"{BASE}/history/{series.path}.json?{q}")
        page = parse_history(payload)
        frames.append(page)
        cursor = payload.get("history.cursor", {}).get("data") or [[0, 0, 100]]
        index, total, size = cursor[0]
        start = index + size
        if start >= total or page.empty:
            break
    return pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
