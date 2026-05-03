# reconstruct_lob.py
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Deque, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

try:
    import orjson as _json

    def _loads(line: bytes) -> dict:
        return _json.loads(line)
except Exception:
    import json as _json

    def _loads(line: bytes) -> dict:
        return _json.loads(line)


EVENT_COLUMNS = ["t_us", "event_type", "side", "level", "price", "size", "mid", "spread"]
SIDE_BID = "bid"
SIDE_ASK = "ask"
EVENT_ADD = "add"
EVENT_CANCEL = "cancel"
EVENT_EXECUTE = "execute"


def _ts_to_us(ts: Any) -> int:
    """
    Convert exchange timestamps to integer microseconds.
    Binance market streams typically publish ms. If already in us, keep as-is.
    """
    x = int(ts)
    # crude but practical: ms timestamps are ~1e12 in 2026, us ~1e15
    return x if x >= 10**14 else x * 1000


def _open_text_or_gzip(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def iter_jsonl(path: str) -> Iterator[dict]:
    with _open_text_or_gzip(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield _loads(line)


def iter_binance_depth(path: str) -> Iterator[dict]:
    """
    Yields raw Binance diff-depth events from JSONL / JSONL.GZ.
    """
    for obj in iter_jsonl(path):
        if obj.get("e") != "depthUpdate":
            continue
        yield obj


def iter_binance_aggtrade(path: str) -> Iterator[dict]:
    """
    Yields raw Binance aggTrade events from JSONL / JSONL.GZ.
    """
    for obj in iter_jsonl(path):
        if obj.get("e") != "aggTrade":
            continue
        yield obj


def merge_exchange_streams(
    depth_events: Iterable[dict],
    trade_events: Iterable[dict],
) -> Iterator[Tuple[str, dict]]:
    """
    Merge two already time-sorted iterables by exchange time.
    Ties are resolved with trades first, then depth, so executions are
    available to classify immediate book reductions.
    """
    depth_it = iter(depth_events)
    trade_it = iter(trade_events)
    heap: List[Tuple[int, int, str, dict]] = []

    def push_next(it: Iterator[dict], kind: str, priority: int) -> None:
        try:
            obj = next(it)
        except StopIteration:
            return
        ts = _ts_to_us(obj.get("T", obj.get("E")))
        heappush(heap, (ts, priority, kind, obj))

    push_next(trade_it, "trade", 0)
    push_next(depth_it, "depth", 1)

    while heap:
        _, _, kind, obj = heappop(heap)
        yield kind, obj
        if kind == "trade":
            push_next(trade_it, "trade", 0)
        else:
            push_next(depth_it, "depth", 1)


@njit(cache=True)
def _binary_search_desc(a: np.ndarray, n: int, x: float) -> int:
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] > x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _binary_search_asc(a: np.ndarray, n: int, x: float) -> int:
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _find_index(a: np.ndarray, n: int, x: float, descending: bool, eps: float) -> int:
    idx = _binary_search_desc(a, n, x) if descending else _binary_search_asc(a, n, x)
    if idx < n and abs(a[idx] - x) <= eps:
        return idx
    return -1


@njit(cache=True)
def _upsert_level(
    prices: np.ndarray,
    qtys: np.ndarray,
    n: int,
    price: float,
    new_qty: float,
    descending: bool,
    eps: float,
) -> Tuple[int, float, int, int, int]:
    """
    In-place upsert/remove on a sorted side array.

    Returns:
        new_n,
        old_qty,
        idx_before,
        idx_after,
        changed_flag (0/1)
    """
    capacity = prices.shape[0]
    idx_after = -1

    ins = _binary_search_desc(prices, n, price) if descending else _binary_search_asc(prices, n, price)
    idx = ins if (ins < n and abs(prices[ins] - price) <= eps) else -1

    if idx != -1:
        old_qty = qtys[idx]
        idx_before = idx

        if new_qty <= eps:
            for j in range(idx, n - 1):
                prices[j] = prices[j + 1]
                qtys[j] = qtys[j + 1]
            prices[n - 1] = 0.0
            qtys[n - 1] = 0.0
            return n - 1, old_qty, idx_before, -1, 1

        if abs(old_qty - new_qty) <= eps:
            return n, old_qty, idx_before, idx, 0

        qtys[idx] = new_qty
        return n, old_qty, idx_before, idx, 1

    # missing level
    old_qty = 0.0
    idx_before = -1

    if new_qty <= eps:
        return n, old_qty, idx_before, -1, 0

    idx_after = ins
    if n < capacity:
        for j in range(n, idx_after, -1):
            prices[j] = prices[j - 1]
            qtys[j] = qtys[j - 1]
        prices[idx_after] = price
        qtys[idx_after] = new_qty
        return n + 1, old_qty, idx_before, idx_after, 1

    # array full: insert only if better than current worst kept level
    if idx_after >= capacity:
        return n, old_qty, idx_before, -1, 0

    for j in range(capacity - 1, idx_after, -1):
        prices[j] = prices[j - 1]
        qtys[j] = qtys[j - 1]
    prices[idx_after] = price
    qtys[idx_after] = new_qty
    return n, old_qty, idx_before, idx_after, 1


@njit(cache=True)
def _best_bid_ask(
    bid_prices: np.ndarray,
    n_bids: int,
    ask_prices: np.ndarray,
    n_asks: int,
) -> Tuple[float, float, float, float]:
    if n_bids <= 0 or n_asks <= 0:
        return np.nan, np.nan, np.nan, np.nan
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    spread = best_ask - best_bid
    mid = 0.5 * (best_bid + best_ask)
    return best_bid, best_ask, mid, spread


@dataclass
class EmittedEvent:
    t_us: int
    event_type: str
    side: str
    level: int
    price: float
    size: float
    mid: float
    spread: float


class TradeMatcher:
    """
    Holds recent aggressive trade flow for classifying negative book deltas.

    Mapping:
      - aggTrade.m == True  => buyer is maker => aggressive SELL => passive BID depleted
      - aggTrade.m == False => buyer not maker => aggressive BUY  => passive ASK depleted
    """

    def __init__(self, match_window_us: int = 250_000, qty_eps: float = 1e-12):
        self.match_window_us = match_window_us
        self.qty_eps = qty_eps
        self.pending: Dict[str, DefaultDict[float, Deque[List[float]]]] = {
            SIDE_BID: defaultdict(deque),
            SIDE_ASK: defaultdict(deque),
        }

    def _prune(self, side: str, price: float, now_us: int) -> None:
        dq = self.pending[side].get(price)
        if not dq:
            return
        while dq and now_us - int(dq[0][0]) > self.match_window_us:
            dq.popleft()
        if not dq:
            self.pending[side].pop(price, None)

    def add_binance_aggtrade(self, trade: dict) -> None:
        t_us = _ts_to_us(trade.get("T", trade["E"]))
        price = float(trade["p"])
        qty = float(trade["q"])
        passive_side = SIDE_BID if bool(trade["m"]) else SIDE_ASK
        dq = self.pending[passive_side][price]
        dq.append([float(t_us), qty])
        self._prune(passive_side, price, t_us)

    def consume(self, side: str, price: float, reduction_qty: float, now_us: int) -> float:
        """
        Return the quantity of a negative depth delta that can be explained by
        recent trades at the same price on the passive side.
        """
        self._prune(side, price, now_us)
        dq = self.pending[side].get(price)
        if not dq:
            return 0.0

        remaining = reduction_qty
        matched = 0.0

        while dq and remaining > self.qty_eps:
            head = dq[0]
            avail = head[1]
            take = avail if avail < remaining else remaining
            matched += take
            remaining -= take
            head[1] -= take
            if head[1] <= self.qty_eps:
                dq.popleft()

        if not dq:
            self.pending[side].pop(price, None)

        return matched


class OrderBookReconstructor:
    """
    Reconstructs the full local L2 book from Binance-style snapshot + diff-depth updates
    and emits only strict top-N event rows.

    Output schema:
      t_us, event_type, side, level, price, size, mid, spread

    Conventions:
      - `size` is the event size (absolute quantity delta), not resting quantity.
      - For add events, `level` is the post-update rank.
      - For cancel/execute events, `level` is the pre-update rank.
      - Only events that affect the visible top-N are emitted.
    """

    def __init__(
        self,
        top_n: int = 10,
        max_levels_per_side: int = 5000,
        qty_eps: float = 1e-12,
        match_window_us: int = 250_000,
        strict_sequence: bool = True,
    ):
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if max_levels_per_side < top_n:
            raise ValueError("max_levels_per_side must be >= top_n")

        self.top_n = top_n
        self.qty_eps = qty_eps
        self.strict_sequence = strict_sequence

        self.bid_prices = np.zeros(max_levels_per_side, dtype=np.float64)
        self.bid_qtys = np.zeros(max_levels_per_side, dtype=np.float64)
        self.ask_prices = np.zeros(max_levels_per_side, dtype=np.float64)
        self.ask_qtys = np.zeros(max_levels_per_side, dtype=np.float64)

        self.n_bids = 0
        self.n_asks = 0

        self.snapshot_last_update_id: Optional[int] = None
        self.prev_stream_u: Optional[int] = None
        self.synced = False

        self.trade_matcher = TradeMatcher(match_window_us=match_window_us, qty_eps=qty_eps)

        # Core emitted columns
        self.out_t_us: List[int] = []
        self.out_event_type: List[str] = []
        self.out_side: List[str] = []
        self.out_level: List[int] = []
        self.out_price: List[float] = []
        self.out_size: List[float] = []
        self.out_mid: List[float] = []
        self.out_spread: List[float] = []

        # Full post-event top-N visible book state columns
        self.out_book_state: Dict[str, List[float]] = {}
        for i in range(1, self.top_n + 1):
            self.out_book_state[f"bid_px_{i}"] = []
            self.out_book_state[f"bid_sz_{i}"] = []
            self.out_book_state[f"ask_px_{i}"] = []
            self.out_book_state[f"ask_sz_{i}"] = []

    # ---------- snapshot / bootstrap ----------

    def load_snapshot(self, snapshot: dict) -> None:
        """
        Load a REST depth snapshot:
          {
            "lastUpdateId": ...,
            "bids": [["price", "qty"], ...],
            "asks": [["price", "qty"], ...]
          }
        """
        self.snapshot_last_update_id = int(snapshot["lastUpdateId"])

        bids = [(float(p), float(q)) for p, q in snapshot["bids"] if float(q) > self.qty_eps]
        asks = [(float(p), float(q)) for p, q in snapshot["asks"] if float(q) > self.qty_eps]

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        self.n_bids = min(len(bids), self.bid_prices.shape[0])
        self.n_asks = min(len(asks), self.ask_prices.shape[0])

        self.bid_prices[:] = 0.0
        self.bid_qtys[:] = 0.0
        self.ask_prices[:] = 0.0
        self.ask_qtys[:] = 0.0

        for i in range(self.n_bids):
            self.bid_prices[i] = bids[i][0]
            self.bid_qtys[i] = bids[i][1]

        for i in range(self.n_asks):
            self.ask_prices[i] = asks[i][0]
            self.ask_qtys[i] = asks[i][1]

        self.prev_stream_u = None
        self.synced = False

    def bootstrap_from_buffer(self, buffered_depth_events: Iterable[dict]) -> None:
        """
        Official sync:
          - drop events with u < snapshot.lastUpdateId
          - first processed event must satisfy U <= lastUpdateId <= u
          - after that, normal sequence checking with pu == previous u
        """
        if self.snapshot_last_update_id is None:
            raise RuntimeError("load_snapshot() must be called before bootstrap_from_buffer()")

        bridged = False
        for ev in buffered_depth_events:
            u = int(ev["u"])
            if u < self.snapshot_last_update_id:
                continue

            U = int(ev["U"])
            if not bridged:
                if U <= self.snapshot_last_update_id <= u:
                    self.synced = True
                    self._apply_depth_event(ev, skip_seq_check=True)
                    bridged = True
                continue

            self.process_depth_event(ev)

        if not bridged:
            raise RuntimeError("No buffered depth event bridged snapshot lastUpdateId")

    # ---------- public processing ----------

    def process_trade_event(self, trade: dict) -> None:
        self.trade_matcher.add_binance_aggtrade(trade)

    def process_depth_event(self, ev: dict) -> None:
        if not self.synced:
            raise RuntimeError("Order book is not synced. Call bootstrap_from_buffer() first.")
        self._apply_depth_event(ev, skip_seq_check=False)

    def process_merged_stream(
        self,
        merged_stream: Iterable[Tuple[str, dict]],
    ) -> None:
        for kind, obj in merged_stream:
            if kind == "trade":
                self.process_trade_event(obj)
            else:
                self.process_depth_event(obj)

    def to_frame(self) -> pd.DataFrame:
        """
        Materialize the emitted event stream with the expanded post-event LOB state.
        """
        data = {
            "t_us": self.out_t_us,
            "event_type": self.out_event_type,
            "side": self.out_side,
            "level": self.out_level,
            "price": self.out_price,
            "size": self.out_size,
            "mid": self.out_mid,
            "spread": self.out_spread,
        }

        for i in range(1, self.top_n + 1):
            data[f"bid_px_{i}"] = self.out_book_state[f"bid_px_{i}"]
            data[f"bid_sz_{i}"] = self.out_book_state[f"bid_sz_{i}"]
            data[f"ask_px_{i}"] = self.out_book_state[f"ask_px_{i}"]
            data[f"ask_sz_{i}"] = self.out_book_state[f"ask_sz_{i}"]

        ordered_cols = [
            "t_us", "event_type", "side", "level", "price", "size", "mid", "spread"
        ]
        for i in range(1, self.top_n + 1):
            ordered_cols.extend([
                f"bid_px_{i}",
                f"bid_sz_{i}",
                f"ask_px_{i}",
                f"ask_sz_{i}",
            ])

        return pd.DataFrame(data, columns=ordered_cols)


    def write_parquet(self, path: str) -> None:
        self.to_frame().to_parquet(path, index=False)
        # ---------- internal ----------

    def _apply_depth_event(self, ev: dict, skip_seq_check: bool) -> None:
        U = int(ev["U"])
        u = int(ev["u"])
        pu = int(ev.get("pu", -1))
        t_us = _ts_to_us(ev.get("T", ev["E"]))

        if self.snapshot_last_update_id is None:
            raise RuntimeError("Snapshot not loaded")

        if not skip_seq_check:
            if self.prev_stream_u is None:
                # defensive check if someone skips bootstrap
                if not (U <= self.snapshot_last_update_id <= u):
                    raise RuntimeError(
                        f"First processed event does not bridge snapshot: "
                        f"U={U}, u={u}, lastUpdateId={self.snapshot_last_update_id}"
                    )
            else:
                if self.strict_sequence and pu != self.prev_stream_u:
                    raise RuntimeError(
                        f"Sequence break: expected pu={self.prev_stream_u}, got pu={pu}"
                    )

        for p, q in ev.get("b", []):
            self._apply_one_level_update(
                side=SIDE_BID,
                price=float(p),
                new_qty=float(q),
                t_us=t_us,
            )

        for p, q in ev.get("a", []):
            self._apply_one_level_update(
                side=SIDE_ASK,
                price=float(p),
                new_qty=float(q),
                t_us=t_us,
            )

        self.prev_stream_u = u

    def _top_n_state_snapshot(self) -> Dict[str, float]:
        """
        Capture the full post-event visible top-N state.
        Missing levels are zero-padded.
        """
        state: Dict[str, float] = {}

        for i in range(self.top_n):
            if i < self.n_bids:
                state[f"bid_px_{i+1}"] = float(self.bid_prices[i])
                state[f"bid_sz_{i+1}"] = float(self.bid_qtys[i])
            else:
                state[f"bid_px_{i+1}"] = 0.0
                state[f"bid_sz_{i+1}"] = 0.0

            if i < self.n_asks:
                state[f"ask_px_{i+1}"] = float(self.ask_prices[i])
                state[f"ask_sz_{i+1}"] = float(self.ask_qtys[i])
            else:
                state[f"ask_px_{i+1}"] = 0.0
                state[f"ask_sz_{i+1}"] = 0.0

        return state

    def _apply_one_level_update(self, side: str, price: float, new_qty: float, t_us: int) -> None:
        is_bid = side == SIDE_BID

        prices = self.bid_prices if is_bid else self.ask_prices
        qtys = self.bid_qtys if is_bid else self.ask_qtys
        n = self.n_bids if is_bid else self.n_asks

        new_n, old_qty, idx_before, idx_after, changed = _upsert_level(
            prices=prices,
            qtys=qtys,
            n=n,
            price=price,
            new_qty=new_qty,
            descending=is_bid,
            eps=self.qty_eps,
        )

        if is_bid:
            self.n_bids = new_n
        else:
            self.n_asks = new_n

        if changed == 0:
            return

        before_rank = idx_before + 1 if idx_before >= 0 else 0
        after_rank = idx_after + 1 if idx_after >= 0 else 0

        affects_top = (0 < before_rank <= self.top_n) or (0 < after_rank <= self.top_n)
        if not affects_top:
            return

        delta = new_qty - old_qty
        if abs(delta) <= self.qty_eps:
            return

        if delta > 0.0:
            level = after_rank
            self._emit(
                EmittedEvent(
                    t_us=t_us,
                    event_type=EVENT_ADD,
                    side=side,
                    level=level,
                    price=price,
                    size=delta,
                    mid=0.0,
                    spread=0.0,
                )
            )
            return

        # negative delta: split into execute + cancel
        reduction = -delta
        level = before_rank if before_rank > 0 else after_rank
        matched_exec = self.trade_matcher.consume(side, price, reduction, t_us)
        residual_cancel = reduction - matched_exec

        if matched_exec > self.qty_eps:
            self._emit(
                EmittedEvent(
                    t_us=t_us,
                    event_type=EVENT_EXECUTE,
                    side=side,
                    level=level,
                    price=price,
                    size=matched_exec,
                    mid=0.0,
                    spread=0.0,
                )
            )

        if residual_cancel > self.qty_eps:
            self._emit(
                EmittedEvent(
                    t_us=t_us,
                    event_type=EVENT_CANCEL,
                    side=side,
                    level=level,
                    price=price,
                    size=residual_cancel,
                    mid=0.0,
                    spread=0.0,
                )
            )

    def _emit(self, ev: EmittedEvent) -> None:
        """
        Emit one event row with the strict schema plus the full post-event top-N LOB state.
        This must be called only AFTER the book update has already been applied.
        """
        if ev.level <= 0 or ev.level > self.top_n:
            return

        _, _, mid, spread = _best_bid_ask(
            self.bid_prices, self.n_bids, self.ask_prices, self.n_asks
        )

        state = self._top_n_state_snapshot()

        self.out_t_us.append(int(ev.t_us))
        self.out_event_type.append(ev.event_type)
        self.out_side.append(ev.side)
        self.out_level.append(int(ev.level))
        self.out_price.append(float(ev.price))
        self.out_size.append(float(ev.size))
        self.out_mid.append(float(mid))
        self.out_spread.append(float(spread))

        for i in range(1, self.top_n + 1):
            self.out_book_state[f"bid_px_{i}"].append(state[f"bid_px_{i}"])
            self.out_book_state[f"bid_sz_{i}"].append(state[f"bid_sz_{i}"])
            self.out_book_state[f"ask_px_{i}"].append(state[f"ask_px_{i}"])
            self.out_book_state[f"ask_sz_{i}"].append(state[f"ask_sz_{i}"])


def load_snapshot_json(path: str) -> dict:
    with _open_text_or_gzip(path) as f:
        raw = f.read()
    return _loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct top-N LOB events from real Binance depth + trades.")
    parser.add_argument("--snapshot", required=True, help="REST snapshot JSON/JSON.GZ")
    parser.add_argument("--depth-buffer", required=True, help="Buffered depth JSONL/JSONL.GZ used for initial sync")
    parser.add_argument("--depth", required=True, help="Post-bootstrap depth JSONL/JSONL.GZ")
    parser.add_argument("--trades", required=True, help="aggTrade JSONL/JSONL.GZ")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-levels", type=int, default=5000)
    parser.add_argument("--match-window-us", type=int, default=250_000)
    args = parser.parse_args()

    snapshot = load_snapshot_json(args.snapshot)

    recon = OrderBookReconstructor(
        top_n=args.top_n,
        max_levels_per_side=args.max_levels,
        match_window_us=args.match_window_us,
        strict_sequence=True,
    )

    recon.load_snapshot(snapshot)
    recon.bootstrap_from_buffer(iter_binance_depth(args.depth_buffer))

    merged = merge_exchange_streams(
        depth_events=iter_binance_depth(args.depth),
        trade_events=iter_binance_aggtrade(args.trades),
    )
    recon.process_merged_stream(merged)
    recon.write_parquet(args.out)


if __name__ == "__main__":
    main()