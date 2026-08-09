"""Stage-5 broker-neutral execution contracts and persisted state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


ALPHAMASTER_MAGIC = 20250101
EXECUTION_STATE_VERSION = "trading-execution-state-v1"
EXECUTION_RECEIPT_VERSION = "execution-receipt-v1"


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _ticket_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} must be a ticket sequence")
    tickets = tuple(value)
    if any(type(ticket) is not int or ticket <= 0 for ticket in tickets):
        raise ValueError(f"{field} contains an invalid ticket")
    if len(set(tickets)) != len(tickets):
        raise ValueError(f"{field} contains duplicate tickets")
    return tickets


@dataclass(frozen=True, slots=True)
class BrokerAccountV1:
    login: int
    server: str
    position_mode: str
    trade_allowed: bool
    trade_expert: bool

    def __post_init__(self) -> None:
        if type(self.login) is not int or self.login <= 0:
            raise ValueError("login must be a positive integer")
        if not self.server:
            raise ValueError("server must not be empty")
        if self.position_mode not in {"hedging", "netting"}:
            raise ValueError("position_mode must be hedging or netting")
        if type(self.trade_allowed) is not bool or type(self.trade_expert) is not bool:
            raise ValueError("trade permissions must be booleans")

    def to_payload(self) -> dict[str, Any]:
        return {
            "login": self.login,
            "server": self.server,
            "position_mode": self.position_mode,
            "trade_allowed": self.trade_allowed,
            "trade_expert": self.trade_expert,
        }


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentV1:
    account: BrokerAccountV1
    symbol: str
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    min_stop_distance: float
    order_mode: int

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if _finite(self.point, field="point") <= 0:
            raise ValueError("point must be positive")
        if type(self.digits) is not int or self.digits < 0:
            raise ValueError("digits must be a non-negative integer")
        minimum = _finite(self.volume_min, field="volume_min")
        maximum = _finite(self.volume_max, field="volume_max")
        step = _finite(self.volume_step, field="volume_step")
        if minimum <= 0 or maximum < minimum or step <= 0:
            raise ValueError("invalid broker volume constraints")
        if _finite(self.min_stop_distance, field="min_stop_distance") < 0:
            raise ValueError("min_stop_distance must be non-negative")
        if type(self.order_mode) is not int or self.order_mode < 0:
            raise ValueError("order_mode must be a non-negative integer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "account": self.account.to_payload(),
            "symbol": self.symbol,
            "point": self.point,
            "digits": self.digits,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "min_stop_distance": self.min_stop_distance,
            "order_mode": self.order_mode,
        }


@dataclass(frozen=True, slots=True)
class BrokerPositionV1:
    ticket: int
    symbol: str
    side: str
    magic: int
    volume: float
    price_open: float
    stop_loss: float
    take_profit: float
    comment: str = ""

    def __post_init__(self) -> None:
        if type(self.ticket) is not int or self.ticket <= 0:
            raise ValueError("position ticket must be positive")
        if self.side not in {"long", "short"}:
            raise ValueError("position side must be long or short")
        if type(self.magic) is not int:
            raise ValueError("position magic must be an integer")
        if _finite(self.volume, field="volume") <= 0:
            raise ValueError("position volume must be positive")
        _finite(self.price_open, field="price_open")
        _finite(self.stop_loss, field="stop_loss")
        _finite(self.take_profit, field="take_profit")

    def to_payload(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "magic": self.magic,
            "volume": self.volume,
            "price_open": self.price_open,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "comment": self.comment,
        }


@dataclass(frozen=True, slots=True)
class BrokerOrderV1:
    ticket: int
    symbol: str
    side: str
    magic: int
    volume: float
    price_open: float
    stop_loss: float
    take_profit: float
    order_type: str
    comment: str = ""

    def __post_init__(self) -> None:
        if type(self.ticket) is not int or self.ticket <= 0:
            raise ValueError("order ticket must be positive")
        if self.side not in {"long", "short"}:
            raise ValueError("order side must be long or short")
        if type(self.magic) is not int:
            raise ValueError("order magic must be an integer")
        if _finite(self.volume, field="volume") <= 0:
            raise ValueError("order volume must be positive")
        for field in ("price_open", "stop_loss", "take_profit"):
            _finite(getattr(self, field), field=field)
        if self.order_type not in {"limit", "stop", "stop_limit"}:
            raise ValueError("unknown pending order type")

    def to_payload(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "magic": self.magic,
            "volume": self.volume,
            "price_open": self.price_open,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "order_type": self.order_type,
            "comment": self.comment,
        }


@dataclass(frozen=True, slots=True)
class BrokerTickV1:
    symbol: str
    bid: float
    ask: float
    point: float

    def __post_init__(self) -> None:
        bid = _finite(self.bid, field="bid")
        ask = _finite(self.ask, field="ask")
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("invalid bid/ask")
        if _finite(self.point, field="point") <= 0:
            raise ValueError("point must be positive")

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class ExecutionReceiptV1:
    action: str
    accepted: bool
    confirmed: bool
    retcode: int | None
    message: str
    order_ticket: int | None = None
    deal_ticket: int | None = None
    position_ticket: int | None = None
    confirmation_kind: str | None = None
    executed_price: float | None = None
    request: tuple[tuple[str, object], ...] = ()
    created_at: float = 0.0
    schema_version: str = EXECUTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("receipt action must not be empty")
        if type(self.accepted) is not bool or type(self.confirmed) is not bool:
            raise ValueError("receipt flags must be booleans")
        if self.confirmed and not self.accepted:
            raise ValueError("an unaccepted request cannot be confirmed")
        for field in ("order_ticket", "deal_ticket", "position_ticket"):
            ticket = getattr(self, field)
            if ticket is not None and (type(ticket) is not int or ticket <= 0):
                raise ValueError(f"{field} must be a positive integer")
        if self.executed_price is not None and _finite(
            self.executed_price, field="executed_price"
        ) <= 0:
            raise ValueError("executed_price must be positive")
        if self.created_at == 0.0:
            object.__setattr__(self, "created_at", time.time())
        _finite(self.created_at, field="created_at")
        if self.schema_version != EXECUTION_RECEIPT_VERSION:
            raise ValueError("unsupported execution receipt version")

    @classmethod
    def ok(
        cls,
        *,
        action: str,
        retcode: int,
        message: str,
        order_ticket: int | None = None,
        deal_ticket: int | None = None,
        position_ticket: int | None = None,
        confirmation_kind: str,
        executed_price: float | None = None,
        request: tuple[tuple[str, object], ...] = (),
    ) -> "ExecutionReceiptV1":
        return cls(
            action=action,
            accepted=True,
            confirmed=True,
            retcode=retcode,
            message=message,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            position_ticket=position_ticket,
            confirmation_kind=confirmation_kind,
            executed_price=executed_price,
            request=request,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "accepted": self.accepted,
            "confirmed": self.confirmed,
            "retcode": self.retcode,
            "message": self.message,
            "order_ticket": self.order_ticket,
            "deal_ticket": self.deal_ticket,
            "position_ticket": self.position_ticket,
            "confirmation_kind": self.confirmation_kind,
            "executed_price": self.executed_price,
            "request": dict(self.request),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ManagedTradeV1:
    decision_id: str
    plan_id: str
    decision_m15_close: int
    config_hash: str
    account_mode: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    tp1_volume: float
    tp2_volume: float
    position_tickets: tuple[int, ...] = ()
    pending_tickets: tuple[int, ...] = ()
    tp1_ticket: int | None = None
    tp2_ticket: int | None = None
    tp1_done: bool = False
    break_even_applied: bool = False

    def __post_init__(self) -> None:
        if not self.decision_id or not self.plan_id:
            raise ValueError("managed trade identity must not be empty")
        if type(self.decision_m15_close) is not int or self.decision_m15_close <= 0:
            raise ValueError("decision_m15_close must be positive")
        if len(self.config_hash) != 64:
            raise ValueError("config_hash must be a SHA-256 hex digest")
        if self.account_mode not in {"hedging", "netting"}:
            raise ValueError("account_mode must be hedging or netting")
        if self.side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        for field in ("entry_price", "stop_loss", "take_profit_1"):
            _finite(getattr(self, field), field=field)
        if self.take_profit_2 is not None:
            _finite(self.take_profit_2, field="take_profit_2")
        if _finite(self.tp1_volume, field="tp1_volume") <= 0:
            raise ValueError("tp1_volume must be positive")
        if _finite(self.tp2_volume, field="tp2_volume") < 0:
            raise ValueError("tp2_volume must be non-negative")
        object.__setattr__(
            self, "position_tickets", _ticket_tuple(self.position_tickets, field="position_tickets")
        )
        object.__setattr__(
            self, "pending_tickets", _ticket_tuple(self.pending_tickets, field="pending_tickets")
        )
        for field in ("tp1_ticket", "tp2_ticket"):
            ticket = getattr(self, field)
            if ticket is not None and (type(ticket) is not int or ticket <= 0):
                raise ValueError(f"{field} must be a positive integer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "plan_id": self.plan_id,
            "decision_m15_close": self.decision_m15_close,
            "config_hash": self.config_hash,
            "account_mode": self.account_mode,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "tp1_volume": self.tp1_volume,
            "tp2_volume": self.tp2_volume,
            "position_tickets": list(self.position_tickets),
            "pending_tickets": list(self.pending_tickets),
            "tp1_ticket": self.tp1_ticket,
            "tp2_ticket": self.tp2_ticket,
            "tp1_done": self.tp1_done,
            "break_even_applied": self.break_even_applied,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ManagedTradeV1":
        return cls(
            decision_id=str(payload["decision_id"]),
            plan_id=str(payload["plan_id"]),
            decision_m15_close=int(payload["decision_m15_close"]),
            config_hash=str(payload["config_hash"]),
            account_mode=str(payload["account_mode"]),
            side=str(payload["side"]),
            entry_price=float(payload["entry_price"]),
            stop_loss=float(payload["stop_loss"]),
            take_profit_1=float(payload["take_profit_1"]),
            take_profit_2=(
                None if payload.get("take_profit_2") is None else float(payload["take_profit_2"])
            ),
            tp1_volume=float(payload["tp1_volume"]),
            tp2_volume=float(payload["tp2_volume"]),
            position_tickets=tuple(payload.get("position_tickets") or ()),
            pending_tickets=tuple(payload.get("pending_tickets") or ()),
            tp1_ticket=payload.get("tp1_ticket"),
            tp2_ticket=payload.get("tp2_ticket"),
            tp1_done=bool(payload.get("tp1_done", False)),
            break_even_applied=bool(payload.get("break_even_applied", False)),
        )


@dataclass(frozen=True, slots=True)
class TradingExecutionStateV1:
    last_processed_m15: int | None = None
    config_hash: str | None = None
    cooldown_until_m15: int | None = None
    managed_trade: ManagedTradeV1 | None = None
    receipt_count: int = 0
    schema_version: str = EXECUTION_STATE_VERSION

    def __post_init__(self) -> None:
        for field in ("last_processed_m15", "cooldown_until_m15"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field} must be a positive integer")
        if self.config_hash is not None and len(self.config_hash) != 64:
            raise ValueError("config_hash must be a SHA-256 hex digest")
        if type(self.receipt_count) is not int or self.receipt_count < 0:
            raise ValueError("receipt_count must be non-negative")
        if self.schema_version != EXECUTION_STATE_VERSION:
            raise ValueError("unsupported execution state version")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_processed_m15": self.last_processed_m15,
            "config_hash": self.config_hash,
            "cooldown_until_m15": self.cooldown_until_m15,
            "managed_trade": None if self.managed_trade is None else self.managed_trade.to_payload(),
            "receipt_count": self.receipt_count,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TradingExecutionStateV1":
        managed_payload = payload.get("managed_trade")
        return cls(
            last_processed_m15=payload.get("last_processed_m15"),
            config_hash=payload.get("config_hash"),
            cooldown_until_m15=payload.get("cooldown_until_m15"),
            managed_trade=(
                None
                if managed_payload is None
                else ManagedTradeV1.from_payload(dict(managed_payload))
            ),
            receipt_count=int(payload.get("receipt_count", 0)),
            schema_version=str(payload.get("schema_version", EXECUTION_STATE_VERSION)),
        )
