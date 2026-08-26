"""Unit tests for analytics.py (bucket aggregation, retention, stockout)."""
from analytics import ShelfAnalytics
from slots import SlotSnapshot


def _snap(slot_id: str, state: str, count: int, by_code: dict, capacity: int = 4):
    return SlotSnapshot(
        slot_id=slot_id, state=state, count=count, by_code=by_code,
        expected_code="A", capacity=capacity,
    )


def _new_analytics(tmp_path, bucket=60, stockout=30, retention_h=24):
    return ShelfAnalytics(
        db_path=str(tmp_path / "test.db"),
        bucket_seconds=bucket,
        retention_hours=retention_h,
        stockout_seconds=stockout,
        alert_cooldown_seconds=1,
    )


def test_bucket_alignment(tmp_path):
    a = _new_analytics(tmp_path, bucket=60)
    events = a.record([_snap("s1", "PARTIAL", 2, {"A": 2})], now=125)
    assert events == []
    hm = a.heatmap(buckets=2)
    assert hm[0]["bucket"] == 120  # floor(125/60)*60
    assert hm[0]["code"] == "A"
    assert hm[0]["max_count"] == 2
    a.close()


def test_peak_upsert_not_cumulative(tmp_path):
    a = _new_analytics(tmp_path, bucket=60)
    a.record([_snap("s1", "PARTIAL", 2, {"A": 2})], now=125)
    a.record([_snap("s1", "FULL", 4, {"A": 4})], now=150)  # same bucket, higher
    hm = a.heatmap(buckets=1)
    row = next(r for r in hm if r["code"] == "A")
    assert row["max_count"] == 4  # peak, not 6
    a.close()


def test_state_ticks_accumulate(tmp_path):
    a = _new_analytics(tmp_path, bucket=600)
    a.record([_snap("s1", "EMPTY", 0, {})], now=1000)
    a.record([_snap("s1", "EMPTY", 0, {})], now=1001)
    a.record([_snap("s1", "PARTIAL", 2, {"A": 2})], now=1002)
    hm = a.heatmap(buckets=1)
    row = next(r for r in hm if r["code"] == "all")
    assert row["state_ticks"]["EMPTY"] == 2
    assert row["state_ticks"]["PARTIAL"] == 1
    assert row["state"] == "EMPTY"  # majority
    a.close()


def test_stockout_after_sustained_empty(tmp_path):
    a = _new_analytics(tmp_path, bucket=600, stockout=30)
    t0 = 1000.0
    evs = a.record([_snap("s1", "EMPTY", 0, {})], now=t0)
    assert evs == []                              # not yet confirmed
    evs = a.record([_snap("s1", "EMPTY", 0, {})], now=t0 + 31)
    assert [e["kind"] for e in evs] == ["stockout"]
    a.close()


def test_restock_after_stockout(tmp_path):
    a = _new_analytics(tmp_path, bucket=600, stockout=30)
    t0 = 1000.0
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0)
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0 + 31)   # stockout fired
    evs = a.record([_snap("s1", "FULL", 4, {"A": 4})], now=t0 + 60)
    assert [e["kind"] for e in evs] == ["restock"]
    a.close()


def test_no_stockout_spam(tmp_path):
    a = _new_analytics(tmp_path, bucket=600, stockout=30)
    t0 = 1000.0
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0)
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0 + 31)   # fires
    evs = a.record([_snap("s1", "EMPTY", 0, {})], now=t0 + 60)
    assert evs == []                                      # no repeat
    a.close()


def test_retention_prunes_old_buckets(tmp_path):
    a = _new_analytics(tmp_path, bucket=100, retention_h=1)
    a.record([_snap("s1", "PARTIAL", 1, {"A": 1})], now=200)
    a.record([_snap("s1", "PARTIAL", 2, {"A": 2})], now=200 + 4000)  # 1h11m later
    hm = a.heatmap(buckets=3)
    buckets = {r["bucket"] for r in hm}
    assert buckets == {4200}  # old 200 pruned by retention; 4200/100*100 aligned
    a.close()


def test_recent_events_order(tmp_path):
    a = _new_analytics(tmp_path, stockout=30)
    t0 = 1000.0
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0)
    a.record([_snap("s1", "EMPTY", 0, {})], now=t0 + 31)
    a.record([_snap("s1", "FULL", 3, {"A": 3})], now=t0 + 50)
    evs = a.recent_events(10)
    assert [e["kind"] for e in evs] == ["stockout", "restock"]
    assert evs[0]["slot_id"] == "s1"
    a.close()