"""File format wrapper for the PPM compressor."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Literal

from .fasta import is_probably_fasta, restore_fasta, transform_fasta, transform_fasta_packed, transform_fasta_repeats

try:
    from ._cppm import compress_payload as _compress_payload_native
    from ._cppm import decompress_payload as _decompress_payload_native
except ImportError:
    _compress_payload_native = None
    _decompress_payload_native = None

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitInput, BitOutput
from .model import EOF_SYMBOL, PPMModel

MAGIC = b"PPMDC"
LEGACY_FORMAT_VERSION = 1
STREAM_FORMAT_VERSION = 2
BLOCK_FORMAT_VERSION = 3
FORMAT_VERSION = BLOCK_FORMAT_VERSION
DEFAULT_ORDER = 5
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024

MODE_RAW = 0
MODE_FASTA = 1
AUTO_SAMPLE_SIZE = 262144

CompressionMode = Literal["auto", "raw", "fasta", "packed", "repeat"]


@dataclass(frozen=True)
class CompressionResult:
    archive: bytes
    mode: str


@dataclass(frozen=True)
class FileCompressionResult:
    input_size: int
    output_size: int
    mode: str


@dataclass(frozen=True)
class FileDecompressionResult:
    input_size: int
    output_size: int


@dataclass(frozen=True)
class PreparedPayload:
    mode: int
    payload: bytes
    fasta_alphabet: bytes = b""


@dataclass(frozen=True)
class Header:
    version: int
    mode: int
    order: int
    fasta_alphabet: bytes = b""


def compress_stream(
    source: BinaryIO,
    target: BinaryIO,
    order: int = DEFAULT_ORDER,
    mode: CompressionMode = "auto",
    block_size: int = DEFAULT_BLOCK_SIZE,
    jobs: int = 0,
) -> str:
    result = compress_bytes(source.read(), order=order, mode=mode, block_size=block_size, jobs=jobs)
    target.write(result.archive)
    return result.mode


def compress_bytes(
    data: bytes,
    order: int = DEFAULT_ORDER,
    mode: CompressionMode = "auto",
    block_size: int = DEFAULT_BLOCK_SIZE,
    jobs: int = 0,
) -> CompressionResult:
    if not 0 <= order <= 16:
        raise ValueError("order must be between 0 and 16")
    _validate_block_size(block_size)

    selected_mode = _select_auto_mode(data, order) if mode == "auto" else mode
    target = bytearray()
    target.extend(MAGIC)
    target.extend(bytes((BLOCK_FORMAT_VERSION, _mode_id(selected_mode), order)))

    blocks = (
        _iter_raw_blocks_from_bytes(data, block_size)
        if selected_mode == "raw"
        else _iter_fasta_blocks_from_bytes(data, block_size)
    )
    _write_block_archive(target, blocks, selected_mode, order, jobs, input_size=len(data), block_size=block_size)
    return CompressionResult(archive=bytes(target), mode=selected_mode)


def decompress_stream(source: BinaryIO, target: BinaryIO) -> int:
    header = _read_header(source)

    if header.version < BLOCK_FORMAT_VERSION:
        return _decompress_legacy_stream(header, source, target)

    written = 0
    while True:
        block_size = _read_u32_optional(source)
        if block_size is None:
            return written

        if header.mode == MODE_FASTA:
            alphabet_size = _read_u16_required(source, "truncated FASTA block alphabet length")
            alphabet = source.read(alphabet_size)
            if len(alphabet) != alphabet_size:
                raise ValueError("truncated FASTA block alphabet")
        else:
            alphabet = b""

        compressed_size = _read_u32_required(source, "truncated compressed block length")
        encoded_payload = source.read(compressed_size)
        if len(encoded_payload) != compressed_size:
            raise ValueError("truncated compressed block")

        decoded_payload = _decode_payload_bytes(encoded_payload, header.order)
        if header.mode == MODE_RAW:
            block = decoded_payload
        else:
            block = restore_fasta(decoded_payload, alphabet)

        if len(block) != block_size:
            raise ValueError("decoded block length does not match archive metadata")

        target.write(block)
        written += len(block)


def compress_file(
    input_path: Path,
    output_path: Path,
    order: int = DEFAULT_ORDER,
    mode: CompressionMode = "auto",
    block_size: int = DEFAULT_BLOCK_SIZE,
    jobs: int = 0,
) -> FileCompressionResult:
    if not 0 <= order <= 16:
        raise ValueError("order must be between 0 and 16")
    _validate_block_size(block_size)

    input_size = input_path.stat().st_size
    selected_mode = _select_auto_mode_file(input_path, order) if mode == "auto" else mode

    with output_path.open("wb") as target:
        target.write(MAGIC)
        target.write(bytes((BLOCK_FORMAT_VERSION, _mode_id(selected_mode), order)))
        if selected_mode == "raw":
            blocks = _iter_raw_blocks_from_file(input_path, block_size)
        else:
            blocks = _iter_fasta_blocks_from_file(input_path, block_size)
        _write_block_archive(target, blocks, selected_mode, order, jobs, input_size=input_size, block_size=block_size)

    return FileCompressionResult(
        input_size=input_size,
        output_size=output_path.stat().st_size,
        mode=selected_mode,
    )


def decompress_file(input_path: Path, output_path: Path) -> FileDecompressionResult:
    with input_path.open("rb") as source, output_path.open("wb") as target:
        written = decompress_stream(source, target)

    return FileDecompressionResult(input_size=input_path.stat().st_size, output_size=written)


def _read_header(source: BinaryIO) -> Header:
    magic = source.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("input is not a ppmdc archive")

    version_raw = source.read(1)
    if len(version_raw) != 1:
        raise ValueError("truncated archive header")

    version = version_raw[0]
    if version == LEGACY_FORMAT_VERSION:
        order_raw = source.read(1)
        if len(order_raw) != 1:
            raise ValueError("truncated archive header")
        return Header(version=version, mode=MODE_RAW, order=order_raw[0])

    if version == STREAM_FORMAT_VERSION:
        return _read_stream_header(source, version)

    if version == BLOCK_FORMAT_VERSION:
        rest = source.read(2)
        if len(rest) != 2:
            raise ValueError("truncated archive header")
        mode = rest[0]
        order = rest[1]
        if mode not in (MODE_RAW, MODE_FASTA):
            raise ValueError(f"unsupported archive mode: {mode}")
        return Header(version=version, mode=mode, order=order)

    raise ValueError(f"unsupported format version: {version}")


def _read_stream_header(source: BinaryIO, version: int) -> Header:
    rest = source.read(2)
    if len(rest) != 2:
        raise ValueError("truncated archive header")

    mode = rest[0]
    order = rest[1]
    if mode == MODE_RAW:
        return Header(version=version, mode=mode, order=order)
    if mode != MODE_FASTA:
        raise ValueError(f"unsupported archive mode: {mode}")

    alphabet_size_raw = source.read(2)
    if len(alphabet_size_raw) != 2:
        raise ValueError("truncated FASTA archive metadata")
    alphabet_size = int.from_bytes(alphabet_size_raw, "big")
    fasta_alphabet = source.read(alphabet_size)
    if len(fasta_alphabet) != alphabet_size:
        raise ValueError("truncated FASTA alphabet")

    return Header(version=version, mode=mode, order=order, fasta_alphabet=fasta_alphabet)


def _decompress_legacy_stream(header: Header, source: BinaryIO, target: BinaryIO) -> int:
    encoded_payload = source.read()
    decoded_payload = _decode_payload_bytes(encoded_payload, header.order)

    if header.mode == MODE_RAW:
        target.write(decoded_payload)
        return len(decoded_payload)

    restored = restore_fasta(decoded_payload, header.fasta_alphabet)
    target.write(restored)
    return len(restored)


def _decode_payload_bytes(payload: bytes, order: int) -> bytes:
    if _decompress_payload_native is not None:
        return _decompress_payload_native(payload, order)

    model = PPMModel(order=order)
    decoder = ArithmeticDecoder(BitInput(BytesReader(payload)))
    decoded = bytearray()

    while True:
        symbol = model.decode_symbol(decoder)
        if symbol == EOF_SYMBOL:
            return bytes(decoded)
        decoded.append(symbol)
        model.update(symbol)


def _compress_payload(payload: bytes, order: int) -> bytes:
    if _compress_payload_native is not None:
        return _compress_payload_native(payload, order)

    target = bytearray()
    model = PPMModel(order=order)
    encoder = ArithmeticEncoder(BitOutput(BytesWriter(target)))
    for value in payload:
        model.encode_symbol(value, encoder)
        model.update(value)

    model.encode_symbol(EOF_SYMBOL, encoder)
    encoder.finish()
    return bytes(target)


def _write_block_archive(
    target: BinaryIO | bytearray,
    blocks: Iterable[bytes],
    mode: Literal["raw", "fasta", "packed", "repeat"],
    order: int,
    jobs: int,
    *,
    input_size: int,
    block_size: int,
) -> None:
    write = target.extend if isinstance(target, bytearray) else target.write
    worker_count = _resolve_jobs(jobs, input_size, block_size)
    tasks = ((mode, order, block) for block in blocks)

    if worker_count == 1:
        results = map(_compress_block_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(_compress_block_task, tasks, chunksize=1)
            for original_size, alphabet, compressed in results:
                _write_block(write, original_size, alphabet, compressed, mode)
            return

    for original_size, alphabet, compressed in results:
        _write_block(write, original_size, alphabet, compressed, mode)


def _write_block(
    write: callable,
    original_size: int,
    alphabet: bytes,
    compressed: bytes,
    mode: Literal["raw", "fasta", "packed", "repeat"],
) -> None:
    write(original_size.to_bytes(4, "big"))
    if mode != "raw":
        write(len(alphabet).to_bytes(2, "big"))
        write(alphabet)
    write(len(compressed).to_bytes(4, "big"))
    write(compressed)


def _compress_block_task(task: tuple[Literal["raw", "fasta", "packed", "repeat"], int, bytes]) -> tuple[int, bytes, bytes]:
    mode, order, block = task
    if mode == "raw":
        return len(block), b"", _compress_payload(block, order)

    if mode == "repeat":
        payload, alphabet = transform_fasta_repeats(block, require_initial_header=False)
    elif mode == "packed":
        payload, alphabet = transform_fasta_packed(block, require_initial_header=False)
    else:
        payload, alphabet = transform_fasta(block, require_initial_header=False)
    return len(block), alphabet, _compress_payload(payload, order)


def _iter_raw_blocks_from_bytes(data: bytes, block_size: int) -> Iterator[bytes]:
    for offset in range(0, len(data), block_size):
        yield data[offset : offset + block_size]


def _iter_fasta_blocks_from_bytes(data: bytes, block_size: int) -> Iterator[bytes]:
    yield from _group_lines_into_blocks(data.splitlines(keepends=True), block_size)


def _iter_raw_blocks_from_file(input_path: Path, block_size: int) -> Iterator[bytes]:
    with input_path.open("rb") as source:
        while True:
            chunk = source.read(block_size)
            if not chunk:
                return
            yield chunk


def _iter_fasta_blocks_from_file(input_path: Path, block_size: int) -> Iterator[bytes]:
    with input_path.open("rb") as source:
        yield from _group_lines_into_blocks(source, block_size)


def _group_lines_into_blocks(lines: Iterable[bytes], block_size: int) -> Iterator[bytes]:
    block = bytearray()
    for line in lines:
        if block and len(block) + len(line) > block_size:
            yield bytes(block)
            block.clear()
        block.extend(line)

    if block:
        yield bytes(block)


def _select_auto_mode(data: bytes, order: int) -> Literal["raw", "fasta", "packed", "repeat"]:
    if not is_probably_fasta(data):
        return "raw"

    sample = data if len(data) <= AUTO_SAMPLE_SIZE else data[:AUTO_SAMPLE_SIZE]
    sizes = {"raw": len(_compress_payload(sample, order))}

    fasta_payload, _ = transform_fasta(sample)
    sizes["fasta"] = len(_compress_payload(fasta_payload, order))

    packed_payload, _ = transform_fasta_packed(sample)
    sizes["packed"] = len(_compress_payload(packed_payload, order))

    repeat_payload, _ = transform_fasta_repeats(sample)
    sizes["repeat"] = len(_compress_payload(repeat_payload, order))

    return min(sizes, key=sizes.__getitem__)


def _select_auto_mode_file(input_path: Path, order: int) -> Literal["raw", "fasta", "packed", "repeat"]:
    with input_path.open("rb") as source:
        sample = source.read(AUTO_SAMPLE_SIZE)
    return _select_auto_mode(sample, order)


def _validate_block_size(block_size: int) -> None:
    if not 1 <= block_size <= (1 << 32) - 1:
        raise ValueError("block_size must be between 1 and 2^32 - 1")


def _resolve_jobs(jobs: int, input_size: int, block_size: int) -> int:
    estimated_blocks = max(1, (input_size + block_size - 1) // block_size)
    if estimated_blocks == 1:
        return 1
    if jobs > 0:
        return min(jobs, estimated_blocks)
    return min(os.cpu_count() or 1, estimated_blocks)


def _mode_name(mode: int) -> str:
    if mode == MODE_RAW:
        return "raw"
    if mode == MODE_FASTA:
        return "fasta"
    raise ValueError(f"unsupported mode id: {mode}")


def _mode_id(mode: Literal["raw", "fasta", "packed", "repeat"]) -> int:
    return MODE_RAW if mode == "raw" else MODE_FASTA


def _read_u32_optional(source: BinaryIO) -> int | None:
    raw = source.read(4)
    if not raw:
        return None
    if len(raw) != 4:
        raise ValueError("truncated block header")
    return int.from_bytes(raw, "big")


def _read_u32_required(source: BinaryIO, message: str) -> int:
    raw = source.read(4)
    if len(raw) != 4:
        raise ValueError(message)
    return int.from_bytes(raw, "big")


def _read_u16_required(source: BinaryIO, message: str) -> int:
    raw = source.read(2)
    if len(raw) != 2:
        raise ValueError(message)
    return int.from_bytes(raw, "big")


class BytesWriter:
    def __init__(self, target: bytearray) -> None:
        self.target = target

    def write(self, chunk: bytes) -> int:
        self.target.extend(chunk)
        return len(chunk)


class BytesReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.data) - self.offset
        chunk = self.data[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk
