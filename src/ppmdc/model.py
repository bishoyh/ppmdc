"""Prediction by partial matching model."""

from __future__ import annotations

from dataclasses import dataclass, field

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder

EOF_SYMBOL = 256
ALPHABET_SIZE = 257
MAX_CONTEXT_TOTAL = 1 << 15


@dataclass(slots=True)
class ContextNode:
    """Counts for a single context."""

    counts: dict[int, int] = field(default_factory=dict)
    children: dict[int, "ContextNode"] = field(default_factory=dict)
    total: int = 0
    symbol_mask: int = 0

    def increment(self, symbol: int) -> None:
        counts = self.counts
        current = counts.get(symbol)
        if current is None:
            counts[symbol] = 1
            self.symbol_mask |= 1 << symbol
        else:
            counts[symbol] = current + 1
        self.total += 1
        if self.total > MAX_CONTEXT_TOTAL:
            self._rescale()

    def _rescale(self) -> None:
        scaled: dict[int, int] = {}
        total = 0
        for symbol, count in self.counts.items():
            scaled_count = (count + 1) // 2
            scaled[symbol] = scaled_count
            total += scaled_count
        self.counts = scaled
        self.total = total


class PPMModel:
    """Adaptive PPM model using a simple escape mechanism and symbol exclusion."""

    __slots__ = ("order", "root", "contexts")

    def __init__(self, order: int) -> None:
        if not 0 <= order <= 16:
            raise ValueError("order must be between 0 and 16")

        self.order = order
        self.root = ContextNode()
        self.contexts: list[ContextNode] = [self.root]

    def encode_symbol(self, symbol: int, encoder: ArithmeticEncoder) -> None:
        excluded_mask = 0

        for node in reversed(self.contexts):
            counts = node.counts
            if not counts:
                continue

            visible_total = 0
            cumulative = 0
            found = False
            symbol_low = 0
            symbol_count = 0

            for candidate in sorted(counts):
                if excluded_mask & (1 << candidate):
                    continue

                count = counts[candidate]
                if candidate == symbol:
                    found = True
                    symbol_low = cumulative
                    symbol_count = count
                cumulative += count
                visible_total += count

            if visible_total == 0:
                excluded_mask |= node.symbol_mask
                continue

            total = visible_total + 1
            if found:
                encoder.update(symbol_low, symbol_low + symbol_count, total)
                return

            encoder.update(visible_total, visible_total + 1, total)
            excluded_mask |= node.symbol_mask

        self._encode_order_minus_one(symbol, excluded_mask, encoder)

    def decode_symbol(self, decoder: ArithmeticDecoder) -> int:
        excluded_mask = 0

        for node in reversed(self.contexts):
            counts = node.counts
            if not counts:
                continue

            visible_total = 0
            for symbol in sorted(counts):
                if excluded_mask & (1 << symbol):
                    continue
                visible_total += counts[symbol]

            if visible_total == 0:
                excluded_mask |= node.symbol_mask
                continue

            total = visible_total + 1
            target = decoder.get_target(total)
            if target == visible_total:
                decoder.update(visible_total, visible_total + 1, total)
                excluded_mask |= node.symbol_mask
                continue

            cumulative = 0
            for symbol in sorted(counts):
                if excluded_mask & (1 << symbol):
                    continue
                count = counts[symbol]
                next_cumulative = cumulative + count
                if target < next_cumulative:
                    decoder.update(cumulative, next_cumulative, total)
                    return symbol
                cumulative = next_cumulative

        return self._decode_order_minus_one(excluded_mask, decoder)

    def update(self, symbol: int) -> None:
        contexts = self.contexts
        for node in contexts:
            node.increment(symbol)

        if not self.order or symbol == EOF_SYMBOL:
            return

        limit = min(len(contexts), self.order)
        new_contexts = [self.root]
        for index in range(limit):
            child = contexts[index].children.setdefault(symbol, ContextNode())
            new_contexts.append(child)
        self.contexts = new_contexts

    @staticmethod
    def _encode_order_minus_one(
        symbol: int,
        excluded_mask: int,
        encoder: ArithmeticEncoder,
    ) -> None:
        excluded_before_symbol = (excluded_mask & ((1 << symbol) - 1)).bit_count()
        index = symbol - excluded_before_symbol
        remaining = ALPHABET_SIZE - excluded_mask.bit_count()
        encoder.update(index, index + 1, remaining)

    @staticmethod
    def _decode_order_minus_one(
        excluded_mask: int,
        decoder: ArithmeticDecoder,
    ) -> int:
        remaining = ALPHABET_SIZE - excluded_mask.bit_count()
        index = decoder.get_target(remaining)
        decoder.update(index, index + 1, remaining)

        for symbol in range(ALPHABET_SIZE):
            if excluded_mask & (1 << symbol):
                continue
            if index == 0:
                return symbol
            index -= 1

        raise AssertionError("order -1 decoding failed")
