import pandas as pd
import pytest

from pulse import build
from pulse.iss import SERIES, fetch, parse_history


def iss_page(rows, index, total, size=100):
    return {"history": {"columns": ["TRADEDATE", "CLOSE"], "data": rows},
            "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[index, total, size]]}}


def fake_get_factory(n=230):
    days = pd.bdate_range("2026-01-05", periods=n)
    rows = [[d.strftime("%Y-%m-%d"), 3000 + i] for i, d in enumerate(days)]
    calls = []

    def get(url):
        start = int(url.split("start=")[1].split("&")[0])
        calls.append(start)
        return iss_page(rows[start:start + 100], start, len(rows))
    return get, calls


def test_parse_history_drops_empty_close():
    df = parse_history(iss_page([["2026-10-01", 2900.5], ["2026-10-02", None]], 0, 2))
    assert list(df.close) == [2900.5]


def test_fetch_paginates():
    get, calls = fake_get_factory(230)
    df = fetch(SERIES[0], "2026-01-01", get=get)
    assert calls == [0, 100, 200]
    assert len(df) == 230 and df.close.iloc[-1] == 3229


def test_kpi():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-12-30", "2026-09-15", "2026-10-15", "2026-10-16"]),
                       "close": [100.0, 110.0, 120.0, 126.0]})
    k = build.kpi(df)
    assert k["d1"] == pytest.approx(0.05)
    assert k["m1"] == pytest.approx(126 / 110 - 1)
    assert k["ytd"] == pytest.approx(0.26)


def test_formatting():
    assert build.fmt_pct(0.0512) == "+5,1%"
    assert build.fmt_pct(-0.2) == "−20,0%"
    assert build.fmt_val(2876.4, "п.") == "2 876"
    assert build.fmt_val(11.234, "₽") == "11,23 ₽"


def test_replace_block():
    text = "intro\n<!-- PULSE:START -->\nold\n<!-- PULSE:END -->\noutro"
    assert build.replace_block(text, "new") == "intro\n<!-- PULSE:START -->\nnew\n<!-- PULSE:END -->\noutro"
    with pytest.raises(ValueError):
        build.replace_block("no markers", "x")


def test_update_keeps_old_data_on_network_error(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "DATA", tmp_path)
    pd.DataFrame({"date": ["2026-10-01"], "close": [1.0]}).to_csv(tmp_path / "IMOEX.csv", index=False)

    def broken(url):
        raise TimeoutError("no network")
    df, err = build.update(SERIES[0], get=broken)
    assert len(df) == 1 and "TimeoutError" in err


def test_first_run_produces_numeric_history(tmp_path, monkeypatch):
    """Первый запуск: локальной истории нет. Цены должны остаться числами, а графики — строиться."""
    monkeypatch.setattr(build, "DATA", tmp_path / "data")
    monkeypatch.setattr(build, "CHARTS", tmp_path / "charts")
    get, _ = fake_get_factory(150)
    df, err = build.update(SERIES[0], get=get)
    assert err is None and len(df) == 150
    assert df.close.dtype.kind == "f" and str(df.date.dtype).startswith("datetime64")
    build.chart_line(df, "test", "t.png")
    assert (tmp_path / "charts" / "t.png").exists()
    again = build.load_local(SERIES[0])
    assert again.close.dtype.kind == "f" and len(again) == 150


def test_split_adjustment():
    dates = pd.bdate_range("2026-04-01", periods=6)
    df = pd.DataFrame({"date": dates, "close": [3200.0, 3180.0, 3196.0, 326.3, 330.0, 333.0]})
    adj, ev = build.adjust_splits(df)
    assert ev == [{"date": dates[3].strftime("%d.%m.%Y"), "kind": "сплит", "ratio": "10:1"}]
    assert adj.close.iloc[2] == pytest.approx(319.6)
    assert (adj.close / adj.close.shift()).dropna().between(0.8, 1.2).all()


def test_reverse_split_snaps_to_round_factor():
    dates = pd.bdate_range("2024-07-10", periods=4)
    df = pd.DataFrame({"date": dates, "close": [0.019, 0.02, 92.95, 93.5]})
    adj, ev = build.adjust_splits(df)
    assert ev[0]["ratio"] == "1:5000" and ev[0]["kind"] == "консолидация"
    assert adj.close.iloc[1] == pytest.approx(100.0)


def test_no_false_splits_on_normal_moves():
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=5), "close": [100.0, 115.0, 92.0, 110.0, 109.0]})
    adj, ev = build.adjust_splits(df)
    assert ev == [] and list(adj.close) == list(df.close)
