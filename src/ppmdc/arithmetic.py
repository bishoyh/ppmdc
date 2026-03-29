"""Bit-level arithmetic coding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3
MASK = FULL_RANGE - 1


class BitOutput:
    """Buffered bit writer."""

    __slots__ = ("stream", "current_byte", "num_bits_filled")

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.current_byte = 0
        self.num_bits_filled = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.stream.write(bytes((self.current_byte,)))
            self.current_byte = 0
            self.num_bits_filled = 0

    def finish(self) -> None:
        if self.num_bits_filled:
            self.current_byte <<= 8 - self.num_bits_filled
            self.stream.write(bytes((self.current_byte,)))
            self.current_byte = 0
            self.num_bits_filled = 0


class BitInput:
    """Buffered bit reader."""

    __slots__ = ("stream", "current_byte", "num_bits_remaining")

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self) -> int:
        if self.num_bits_remaining == 0:
            chunk = self.stream.read(1)
            if not chunk:
                return 0
            self.current_byte = chunk[0]
            self.num_bits_remaining = 8

        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1


@dataclass(slots=True)
class ArithmeticEncoder:
    """Finite-precision arithmetic encoder."""

    output: BitOutput
    low: int = 0
    high: int = MASK
    pending_bits: int = 0

    def update(self, cumulative_low: int, cumulative_high: int, total: int) -> None:
        if not 0 <= cumulative_low < cumulative_high <= total:
            raise ValueError("invalid cumulative frequency range")
        if total <= 0:
            raise ValueError("total frequency must be positive")

        current_range = self.high - self.low + 1
        self.high = self.low + (current_range * cumulative_high // total) - 1
        self.low = self.low + (current_range * cumulative_low // total)

        while True:
            if self.high < HALF_RANGE:
                self._shift(0)
            elif self.low >= HALF_RANGE:
                self._shift(1)
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.pending_bits += 1
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1

    def finish(self) -> None:
        self.pending_bits += 1
        if self.low < QUARTER_RANGE:
            self._shift(0)
        else:
            self._shift(1)
        self.output.finish()

    def _shift(self, bit: int) -> None:
        self.output.write(bit)
        opposite = bit ^ 1
        while self.pending_bits:
            self.output.write(opposite)
            self.pending_bits -= 1


@dataclass(slots=True)
class ArithmeticDecoder:
    """Finite-precision arithmetic decoder."""

    input_bits: BitInput
    low: int = 0
    high: int = MASK
    code: int = 0

    def __post_init__(self) -> None:
        for _ in range(STATE_BITS):
            self.code = ((self.code << 1) & MASK) | self.input_bits.read()

    def get_target(self, total: int) -> int:
        if total <= 0:
            raise ValueError("total frequency must be positive")
        current_range = self.high - self.low + 1
        offset = self.code - self.low
        return ((offset + 1) * total - 1) // current_range

    def update(self, cumulative_low: int, cumulative_high: int, total: int) -> None:
        if not 0 <= cumulative_low < cumulative_high <= total:
            raise ValueError("invalid cumulative frequency range")
        current_range = self.high - self.low + 1
        self.high = self.low + (current_range * cumulative_high // total) - 1
        self.low = self.low + (current_range * cumulative_low // total)

        while True:
            if self.high < HALF_RANGE:
                pass
            elif self.low >= HALF_RANGE:
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
                self.code -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
                self.code -= QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
            self.code = ((self.code << 1) & MASK) | self.input_bits.read()
