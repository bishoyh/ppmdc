"""FASTA-aware transform helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

try:
    from ._cppm import build_kmer_signature as _build_kmer_signature_native
    from ._cppm import restore_fasta as _restore_fasta_native
    from ._cppm import transform_fasta as _transform_fasta_native
except ImportError:
    _build_kmer_signature_native = None
    _restore_fasta_native = None
    _transform_fasta_native = None

HEADER_TAG = 0
SEQUENCE_TAG = 1
SEQUENCE_REPEAT_TAG = 2
SEQUENCE_GROUP_TAG = 3
SEQUENCE_PACKED_TAG = 4
SEQUENCE_GROUP_PACKED_TAG = 5

LITERAL_RUN_TAG = 0
MATCH_RUN_TAG = 1
REVERSE_COMPLEMENT_RUN_TAG = 2

NEWLINE_NONE = 0
NEWLINE_LF = 1
NEWLINE_CRLF = 2
NEWLINE_CR = 3

KMER_SIGNATURE_MAGIC = b"KMS1"
PROBABLE_SEQUENCE_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*.-"
)
REPEAT_KMER_SIZE = 8
REPEAT_MIN_MATCH = 12
REPEAT_MAX_CANDIDATES = 32

IUPAC_COMPLEMENTS = {
    ord("A"): ord("T"),
    ord("B"): ord("V"),
    ord("C"): ord("G"),
    ord("D"): ord("H"),
    ord("G"): ord("C"),
    ord("H"): ord("D"),
    ord("K"): ord("M"),
    ord("M"): ord("K"),
    ord("N"): ord("N"),
    ord("R"): ord("Y"),
    ord("S"): ord("S"),
    ord("T"): ord("A"),
    ord("U"): ord("A"),
    ord("V"): ord("B"),
    ord("W"): ord("W"),
    ord("X"): ord("X"),
    ord("Y"): ord("R"),
    ord("-"): ord("-"),
    ord("."): ord("."),
    ord("*"): ord("*"),
}
IUPAC_COMPLEMENTS.update({ord(chr(byte).lower()): ord(chr(complement).lower()) for byte, complement in list(IUPAC_COMPLEMENTS.items()) if 65 <= byte <= 90})


@dataclass(frozen=True)
class FastaLine:
    """One FASTA line without its line terminator."""

    is_header: bool
    body: bytes
    newline: int


def is_probably_fasta(data: bytes) -> bool:
    """Heuristic auto-detection for FASTA-like inputs."""

    try:
        lines = parse_fasta_lines(data)
    except ValueError:
        return False

    saw_sequence = False
    for line in lines:
        if line.is_header:
            continue
        saw_sequence = True
        if any(byte not in PROBABLE_SEQUENCE_BYTES for byte in line.body):
            return False

    return saw_sequence


def transform_fasta(data: bytes, *, require_initial_header: bool = True) -> tuple[bytes, bytes]:
    """Transform FASTA bytes into a structured payload and alphabet metadata."""

    if _transform_fasta_native is not None:
        return _transform_fasta_native(data, False, require_initial_header, False)

    return _transform_fasta_python(data, require_initial_header=require_initial_header, packed=False)


def transform_fasta_packed(data: bytes, *, require_initial_header: bool = True) -> tuple[bytes, bytes]:
    """Transform FASTA bytes into a packed 2-bit payload when the alphabet permits it."""

    if _transform_fasta_native is not None:
        return _transform_fasta_native(data, False, require_initial_header, True)

    return _transform_fasta_python(data, require_initial_header=require_initial_header, packed=True)


def build_kmer_signature(
    data: bytes,
    *,
    kmer_size: int = 15,
    sample_rate: int = 8,
) -> bytes:
    """Build a canonical FASTA k-mer signature for similarity-oriented compression."""

    if _build_kmer_signature_native is not None:
        return _build_kmer_signature_native(data, kmer_size, sample_rate)

    if not 1 <= kmer_size <= 31:
        raise ValueError("kmer_size must be between 1 and 31")
    if sample_rate < 1:
        raise ValueError("sample_rate must be at least 1")

    lines = parse_fasta_lines(data)
    mask = (1 << (2 * kmer_size)) - 1
    shift = 2 * (kmer_size - 1)
    forward = 0
    reverse = 0
    valid = 0
    kmers: set[int] = set()

    for line in lines:
        if line.is_header:
            forward = 0
            reverse = 0
            valid = 0
            continue

        for byte in line.body:
            code = _dna_base_code(byte)
            if code < 0:
                forward = 0
                reverse = 0
                valid = 0
                continue

            forward = ((forward << 2) | code) & mask
            reverse = (reverse >> 2) | ((code ^ 0x03) << shift)
            valid = min(valid + 1, kmer_size)
            if valid < kmer_size:
                continue

            canonical = min(forward, reverse)
            if sample_rate > 1 and _splitmix64(canonical) % sample_rate != 0:
                continue
            kmers.add(canonical)

    encoded = bytearray(KMER_SIGNATURE_MAGIC)
    encoded.extend(_encode_varint(kmer_size))
    encoded.extend(_encode_varint(sample_rate))
    encoded.extend(_encode_varint(len(kmers)))
    previous = 0
    for canonical in sorted(kmers):
        encoded.extend(_encode_varint(canonical - previous))
        previous = canonical
    return bytes(encoded)


def merge_kmer_signatures(left: bytes, right: bytes) -> bytes:
    """Merge two canonical k-mer signatures into a union signature."""

    left_kmer_size, left_sample_rate, _, _ = _parse_kmer_signature(left)
    right_kmer_size, right_sample_rate, _, _ = _parse_kmer_signature(right)
    if left_kmer_size != right_kmer_size or left_sample_rate != right_sample_rate:
        raise ValueError("k-mer signatures must use the same parameters to merge")

    merged_values = bytearray(KMER_SIGNATURE_MAGIC)
    merged_values.extend(_encode_varint(left_kmer_size))
    merged_values.extend(_encode_varint(left_sample_rate))

    merged_count = 0
    previous = 0
    payload = bytearray()
    for canonical in _iter_merged_kmer_values(left, right):
        payload.extend(_encode_varint(canonical - previous))
        previous = canonical
        merged_count += 1

    merged_values.extend(_encode_varint(merged_count))
    merged_values.extend(payload)
    return bytes(merged_values)


def _transform_fasta_python(
    data: bytes,
    *,
    require_initial_header: bool,
    packed: bool,
) -> tuple[bytes, bytes]:
    lines = parse_fasta_lines(data, require_initial_header=require_initial_header)
    alphabet = _sequence_alphabet(lines)
    symbol_map = {byte: index for index, byte in enumerate(alphabet)}
    packed = packed and len(alphabet) <= 4

    payload = bytearray()
    for line in lines:
        if line.is_header:
            payload.append(HEADER_TAG)
            payload.extend(_encode_varint(len(line.body)))
            payload.extend(line.body)
            payload.append(line.newline)
            continue

        payload.append(SEQUENCE_PACKED_TAG if packed else SEQUENCE_TAG)
        payload.extend(_encode_varint(len(line.body)))
        encoded = bytes(symbol_map[byte] for byte in line.body)
        if packed:
            payload.extend(_pack_2bit(encoded))
        else:
            payload.extend(encoded)
        payload.append(line.newline)

    return bytes(payload), alphabet


def transform_fasta_repeats(data: bytes, *, require_initial_header: bool = True) -> tuple[bytes, bytes]:
    """Transform FASTA bytes into a payload with repeat-aware sequence tokens."""

    if _transform_fasta_native is not None:
        return _transform_fasta_native(data, True, require_initial_header, True)

    lines = parse_fasta_lines(data, require_initial_header=require_initial_header)
    alphabet = _sequence_alphabet(lines)
    symbol_map = {byte: index for index, byte in enumerate(alphabet)}
    complement_map = _build_complement_map(alphabet)
    forward_index: dict[bytes, deque[int]] = {}
    reverse_index: dict[bytes, deque[int]] = {}
    history = bytearray()

    payload = bytearray()
    offset = 0
    while offset < len(lines):
        line = lines[offset]
        if line.is_header:
            payload.append(HEADER_TAG)
            payload.extend(_encode_varint(len(line.body)))
            payload.extend(line.body)
            payload.append(line.newline)
            offset += 1
            continue

        group_lines: list[FastaLine] = []
        while offset < len(lines) and not lines[offset].is_header:
            group_lines.append(lines[offset])
            offset += 1

        encoded_lines = [bytes(symbol_map[byte] for byte in group_line.body) for group_line in group_lines]
        group_sequence = b"".join(encoded_lines)
        encoded_body = _encode_repeat_body(
            group_sequence,
            history,
            forward_index,
            reverse_index,
            complement_map,
        )
        if encoded_body is None:
            encoded_body = _encode_literal_repeat_body(group_sequence)

        raw_size = sum(
            1 + len(_encode_varint(len(encoded_line))) + len(encoded_line) + 1
            for encoded_line in encoded_lines
        )
        group_size = (
            1
            + len(_encode_varint(len(group_lines)))
            + sum(len(_encode_varint(len(encoded_line))) + 1 for encoded_line in encoded_lines)
            + len(_encode_varint(len(encoded_body)))
            + len(encoded_body)
        )

        if group_size < raw_size:
            payload.append(SEQUENCE_GROUP_TAG)
            payload.extend(_encode_varint(len(group_lines)))
            for group_line in group_lines:
                payload.extend(_encode_varint(len(group_line.body)))
                payload.append(group_line.newline)
            payload.extend(_encode_varint(len(encoded_body)))
            payload.extend(encoded_body)
        else:
            for encoded_line, group_line in zip(encoded_lines, group_lines):
                payload.append(SEQUENCE_TAG)
                payload.extend(_encode_varint(len(encoded_line)))
                payload.extend(encoded_line)
                payload.append(group_line.newline)

        _index_sequence(group_sequence, history, forward_index, reverse_index, complement_map)

    return bytes(payload), alphabet


def restore_fasta(payload: bytes, alphabet: bytes) -> bytes:
    """Restore original FASTA bytes from a transformed payload."""

    if _restore_fasta_native is not None:
        return _restore_fasta_native(payload, alphabet)

    output = bytearray()
    offset = 0
    history = bytearray()
    complement_map = _build_complement_map(alphabet)

    while offset < len(payload):
        tag = payload[offset]
        offset += 1
        length, offset = _decode_varint(payload, offset)

        if tag == HEADER_TAG:
            if offset + length + 1 > len(payload):
                raise ValueError("truncated FASTA payload")
            body = payload[offset : offset + length]
            offset += length
            newline = payload[offset]
            offset += 1
            output.extend(b">")
            output.extend(body)
            output.extend(_decode_newline(newline))
            continue

        if tag == SEQUENCE_TAG:
            if offset + length + 1 > len(payload):
                raise ValueError("truncated FASTA payload")
            body = payload[offset : offset + length]
            offset += length
            newline = payload[offset]
            offset += 1

            for index in body:
                if index >= len(alphabet):
                    raise ValueError("FASTA payload references an unknown alphabet symbol")
                output.append(alphabet[index])
            history.extend(body)
            output.extend(_decode_newline(newline))
            continue

        if tag == SEQUENCE_PACKED_TAG:
            packed_length = (length + 3) // 4
            if offset + packed_length + 1 > len(payload):
                raise ValueError("truncated FASTA packed payload")
            body = _unpack_2bit(payload[offset : offset + packed_length], length)
            offset += packed_length
            newline = payload[offset]
            offset += 1

            for index in body:
                if index >= len(alphabet):
                    raise ValueError("FASTA payload references an unknown alphabet symbol")
                output.append(alphabet[index])
            history.extend(body)
            output.extend(_decode_newline(newline))
            continue

        if tag == SEQUENCE_REPEAT_TAG:
            encoded_length, offset = _decode_varint(payload, offset)
            if offset + encoded_length + 1 > len(payload):
                raise ValueError("truncated FASTA repeat payload")
            encoded_body = payload[offset : offset + encoded_length]
            offset += encoded_length
            newline = payload[offset]
            offset += 1

            body = _decode_repeat_body(encoded_body, length, history, complement_map)
            for index in body:
                if index >= len(alphabet):
                    raise ValueError("FASTA payload references an unknown alphabet symbol")
                output.append(alphabet[index])
            history.extend(body)
            output.extend(_decode_newline(newline))
            continue

        if tag == SEQUENCE_GROUP_TAG:
            line_count = length
            line_lengths: list[int] = []
            newlines: list[int] = []
            for _ in range(line_count):
                line_length, offset = _decode_varint(payload, offset)
                if offset >= len(payload):
                    raise ValueError("truncated FASTA sequence-group metadata")
                line_lengths.append(line_length)
                newlines.append(payload[offset])
                offset += 1

            encoded_length, offset = _decode_varint(payload, offset)
            if offset + encoded_length > len(payload):
                raise ValueError("truncated FASTA sequence-group payload")
            encoded_body = payload[offset : offset + encoded_length]
            offset += encoded_length

            body = _decode_repeat_body(encoded_body, sum(line_lengths), history, complement_map)
            body_offset = 0
            for line_length, newline in zip(line_lengths, newlines):
                line_body = body[body_offset : body_offset + line_length]
                body_offset += line_length
                for index in line_body:
                    if index >= len(alphabet):
                        raise ValueError("FASTA payload references an unknown alphabet symbol")
                    output.append(alphabet[index])
                output.extend(_decode_newline(newline))
            history.extend(body)
            continue

        if tag == SEQUENCE_GROUP_PACKED_TAG:
            line_count = length
            line_lengths: list[int] = []
            newlines: list[int] = []
            total_length = 0
            for _ in range(line_count):
                line_length, offset = _decode_varint(payload, offset)
                if offset >= len(payload):
                    raise ValueError("truncated FASTA sequence-group metadata")
                line_lengths.append(line_length)
                newlines.append(payload[offset])
                offset += 1
                total_length += line_length

            packed_length = (total_length + 3) // 4
            if offset + packed_length > len(payload):
                raise ValueError("truncated FASTA packed sequence-group payload")
            body = _unpack_2bit(payload[offset : offset + packed_length], total_length)
            offset += packed_length

            body_offset = 0
            for line_length, newline in zip(line_lengths, newlines):
                line_body = body[body_offset : body_offset + line_length]
                body_offset += line_length
                for index in line_body:
                    if index >= len(alphabet):
                        raise ValueError("FASTA payload references an unknown alphabet symbol")
                    output.append(alphabet[index])
                output.extend(_decode_newline(newline))
            history.extend(body)
            continue

        if tag != SEQUENCE_TAG:
            raise ValueError(f"unknown FASTA payload tag: {tag}")

    return bytes(output)


def parse_fasta_lines(data: bytes, *, require_initial_header: bool = True) -> list[FastaLine]:
    """Parse a FASTA-like file into exact header and sequence lines."""

    if not data:
        raise ValueError("empty input is not FASTA")

    lines: list[FastaLine] = []
    offset = 0
    saw_header = not require_initial_header

    while offset < len(data):
        line, newline, offset = _read_line(data, offset)
        is_header = line.startswith(b">")

        if not saw_header:
            if not is_header:
                raise ValueError("FASTA input must begin with a header line")
            saw_header = True

        if is_header:
            lines.append(FastaLine(is_header=True, body=line[1:], newline=newline))
        else:
            lines.append(FastaLine(is_header=False, body=line, newline=newline))

    return lines


def _sequence_alphabet(lines: Iterable[FastaLine]) -> bytes:
    return bytes(
        sorted(
            {
                byte
                for line in lines
                if not line.is_header
                for byte in line.body
            }
        )
    )


def _encode_repeat_body(
    sequence: bytes,
    history: bytearray,
    forward_index: dict[bytes, deque[int]],
    reverse_index: dict[bytes, deque[int]],
    complement_map: list[int],
) -> bytes | None:
    if len(sequence) < REPEAT_MIN_MATCH:
        return None

    local_forward: dict[bytes, deque[int]] = {}
    local_reverse: dict[bytes, deque[int]] = {}
    tokens = bytearray()
    literal_start = 0
    position = 0
    indexed_prefix = 0

    while position < len(sequence):
        indexed_prefix = _index_local_prefix(
            sequence,
            indexed_prefix,
            position,
            len(history),
            local_forward,
            local_reverse,
            complement_map,
        )
        match = _find_best_repeat(
            sequence,
            position,
            history,
            forward_index,
            reverse_index,
            local_forward,
            local_reverse,
            complement_map,
        )
        if match is None:
            position += 1
            continue

        token, match_length, distance = match
        if literal_start < position:
            literal = sequence[literal_start:position]
            tokens.append(LITERAL_RUN_TAG)
            tokens.extend(_encode_varint(len(literal)))
            tokens.extend(literal)

        tokens.append(token)
        tokens.extend(_encode_varint(match_length))
        tokens.extend(_encode_varint(distance))
        position += match_length
        literal_start = position

    if literal_start == 0:
        return None

    if literal_start < len(sequence):
        literal = sequence[literal_start:]
        tokens.append(LITERAL_RUN_TAG)
        tokens.extend(_encode_varint(len(literal)))
        tokens.extend(literal)

    return bytes(tokens)


def _encode_literal_repeat_body(sequence: bytes) -> bytes:
    tokens = bytearray()
    tokens.append(LITERAL_RUN_TAG)
    tokens.extend(_encode_varint(len(sequence)))
    tokens.extend(sequence)
    return bytes(tokens)


def _find_best_repeat(
    sequence: bytes,
    position: int,
    history: bytearray,
    forward_index: dict[bytes, deque[int]],
    reverse_index: dict[bytes, deque[int]],
    local_forward: dict[bytes, deque[int]],
    local_reverse: dict[bytes, deque[int]],
    complement_map: list[int],
) -> tuple[int, int, int] | None:
    if position + REPEAT_KMER_SIZE > len(sequence):
        return None

    key = sequence[position : position + REPEAT_KMER_SIZE]
    current_reference_length = len(history) + position
    best_token = -1
    best_length = 0
    best_distance = 0

    for candidate_index in (forward_index, local_forward):
        for source_start in reversed(candidate_index.get(key, ())):
            length = _forward_repeat_length(sequence, position, source_start, history)
            if length > best_length:
                best_token = MATCH_RUN_TAG
                best_length = length
                best_distance = current_reference_length - source_start

    for candidate_index in (reverse_index, local_reverse):
        for source_end in reversed(candidate_index.get(key, ())):
            length = _reverse_repeat_length(sequence, position, source_end, history, complement_map)
            if length > best_length:
                best_token = REVERSE_COMPLEMENT_RUN_TAG
                best_length = length
                best_distance = current_reference_length - 1 - source_end

    if best_length < REPEAT_MIN_MATCH:
        return None

    return best_token, best_length, best_distance


def _decode_repeat_body(
    encoded_body: bytes,
    expected_length: int,
    history: bytearray,
    complement_map: list[int],
) -> bytes:
    output = bytearray()
    offset = 0

    while offset < len(encoded_body):
        token = encoded_body[offset]
        offset += 1
        length, offset = _decode_varint(encoded_body, offset)

        if token == LITERAL_RUN_TAG:
            if offset + length > len(encoded_body):
                raise ValueError("truncated FASTA repeat literal")
            output.extend(encoded_body[offset : offset + length])
            offset += length
            continue

        distance, offset = _decode_varint(encoded_body, offset)
        if token == MATCH_RUN_TAG:
            source_start = len(history) + len(output) - distance
            if source_start < 0:
                raise ValueError("invalid FASTA repeat match")
            for copied in range(length):
                output.append(_read_reference_byte(history, output, source_start + copied))
            continue

        if token == REVERSE_COMPLEMENT_RUN_TAG:
            source_end = len(history) + len(output) - 1 - distance
            if source_end < 0 or source_end - length + 1 < 0:
                raise ValueError("invalid FASTA reverse-complement match")
            for position in range(source_end, source_end - length, -1):
                output.append(complement_map[_read_reference_byte(history, output, position)])
            continue

        raise ValueError(f"unknown FASTA repeat token: {token}")

    if len(output) != expected_length:
        raise ValueError("decoded FASTA repeat length does not match metadata")

    return bytes(output)


def _build_complement_map(alphabet: bytes) -> list[int]:
    index_by_byte = {byte: index for index, byte in enumerate(alphabet)}
    complement_map: list[int] = []
    for byte in alphabet:
        complement_byte = IUPAC_COMPLEMENTS.get(byte, byte)
        complement_map.append(index_by_byte.get(complement_byte, index_by_byte[byte]))
    return complement_map


def _index_sequence(
    sequence: bytes,
    history: bytearray,
    forward_index: dict[bytes, deque[int]],
    reverse_index: dict[bytes, deque[int]],
    complement_map: list[int],
) -> None:
    start = len(history)
    history.extend(sequence)
    if len(history) < REPEAT_KMER_SIZE:
        return

    first_end = max(REPEAT_KMER_SIZE - 1, start)
    for end in range(first_end, len(history)):
        window_start = end - REPEAT_KMER_SIZE + 1
        _append_kmer(forward_index, bytes(history[window_start : end + 1]), window_start)
        reverse_key = _reverse_complement_key(history, end, complement_map)
        _append_kmer(reverse_index, reverse_key, end)


def _append_kmer(index: dict[bytes, deque[int]], key: bytes, position: int) -> None:
    bucket = index.get(key)
    if bucket is None:
        bucket = deque(maxlen=REPEAT_MAX_CANDIDATES)
        index[key] = bucket
    bucket.append(position)


def _reverse_complement_key(history: bytearray, end: int, complement_map: list[int]) -> bytes:
    return bytes(complement_map[history[position]] for position in range(end, end - REPEAT_KMER_SIZE, -1))


def _index_local_prefix(
    sequence: bytes,
    indexed_prefix: int,
    prefix_length: int,
    history_length: int,
    local_forward: dict[bytes, deque[int]],
    local_reverse: dict[bytes, deque[int]],
    complement_map: list[int],
) -> int:
    if prefix_length < REPEAT_KMER_SIZE:
        return prefix_length

    for end in range(max(REPEAT_KMER_SIZE - 1, indexed_prefix), prefix_length):
        window_start = end - REPEAT_KMER_SIZE + 1
        _append_kmer(local_forward, bytes(sequence[window_start : end + 1]), history_length + window_start)
        reverse_key = bytes(
            complement_map[sequence[position]]
            for position in range(end, end - REPEAT_KMER_SIZE, -1)
        )
        _append_kmer(local_reverse, reverse_key, history_length + end)
    return prefix_length


def _forward_repeat_length(
    sequence: bytes,
    position: int,
    source_start: int,
    history: bytearray,
) -> int:
    length = REPEAT_KMER_SIZE
    max_length = len(sequence) - position
    while length < max_length:
        if _read_reference_sequence_byte(history, sequence, source_start + length) != sequence[position + length]:
            break
        length += 1
    return length


def _reverse_repeat_length(
    sequence: bytes,
    position: int,
    source_end: int,
    history: bytearray,
    complement_map: list[int],
) -> int:
    length = REPEAT_KMER_SIZE
    max_length = min(len(sequence) - position, source_end + 1)
    while length < max_length:
        complement = complement_map[_read_reference_sequence_byte(history, sequence, source_end - length)]
        if complement != sequence[position + length]:
            break
        length += 1
    return length


def _read_reference_sequence_byte(history: bytearray, sequence: bytes, index: int) -> int:
    if index < len(history):
        return history[index]
    return sequence[index - len(history)]


def _read_reference_byte(history: bytearray, output: bytearray, index: int) -> int:
    if index < len(history):
        return history[index]
    output_index = index - len(history)
    if output_index < 0 or output_index >= len(output):
        raise ValueError("repeat match points outside decoded prefix")
    return output[output_index]


def _read_line(data: bytes, offset: int) -> tuple[bytes, int, int]:
    start = offset
    end = len(data)

    while offset < end:
        current = data[offset]
        if current == 0x0A:
            return data[start:offset], NEWLINE_LF, offset + 1
        if current == 0x0D:
            if offset + 1 < end and data[offset + 1] == 0x0A:
                return data[start:offset], NEWLINE_CRLF, offset + 2
            return data[start:offset], NEWLINE_CR, offset + 1
        offset += 1

    return data[start:end], NEWLINE_NONE, end


def _parse_kmer_signature(signature: bytes) -> tuple[int, int, int, int]:
    if not signature.startswith(KMER_SIGNATURE_MAGIC):
        raise ValueError("invalid k-mer signature magic")

    offset = len(KMER_SIGNATURE_MAGIC)
    kmer_size, offset = _decode_varint(signature, offset)
    sample_rate, offset = _decode_varint(signature, offset)
    count, offset = _decode_varint(signature, offset)
    return kmer_size, sample_rate, count, offset


def _iter_kmer_signature_values(signature: bytes) -> Iterable[int]:
    _, _, count, offset = _parse_kmer_signature(signature)
    current = 0
    for _ in range(count):
        delta, offset = _decode_varint(signature, offset)
        current += delta
        yield current

    if offset != len(signature):
        raise ValueError("unexpected trailing bytes in k-mer signature")


def _iter_merged_kmer_values(left: bytes, right: bytes) -> Iterable[int]:
    left_iter = iter(_iter_kmer_signature_values(left))
    right_iter = iter(_iter_kmer_signature_values(right))

    left_value = next(left_iter, None)
    right_value = next(right_iter, None)
    while left_value is not None or right_value is not None:
        if right_value is None or (left_value is not None and left_value < right_value):
            yield left_value
            left_value = next(left_iter, None)
            continue
        if left_value is None or right_value < left_value:
            yield right_value
            right_value = next(right_iter, None)
            continue

        yield left_value
        left_value = next(left_iter, None)
        right_value = next(right_iter, None)


def _dna_base_code(byte: int) -> int:
    if byte in (ord("A"), ord("a")):
        return 0
    if byte in (ord("C"), ord("c")):
        return 1
    if byte in (ord("G"), ord("g")):
        return 2
    if byte in (ord("T"), ord("t")):
        return 3
    return -1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode negative values")

    encoded = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            encoded.append(chunk | 0x80)
        else:
            encoded.append(chunk)
            return bytes(encoded)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0

    while True:
        if offset >= len(data):
            raise ValueError("truncated varint")

        current = data[offset]
        offset += 1
        value |= (current & 0x7F) << shift

        if not current & 0x80:
            return value, offset

        shift += 7
        if shift > 63:
            raise ValueError("varint is too large")


def _decode_newline(newline: int) -> bytes:
    if newline == NEWLINE_NONE:
        return b""
    if newline == NEWLINE_LF:
        return b"\n"
    if newline == NEWLINE_CRLF:
        return b"\r\n"
    if newline == NEWLINE_CR:
        return b"\r"
    raise ValueError(f"unknown newline code: {newline}")


def _pack_2bit(symbols: bytes) -> bytes:
    packed = bytearray()
    for offset in range(0, len(symbols), 4):
        chunk = symbols[offset : offset + 4]
        value = 0
        for symbol in chunk:
            value = (value << 2) | symbol
        value <<= 2 * (4 - len(chunk))
        packed.append(value)
    return bytes(packed)


def _unpack_2bit(packed: bytes, symbol_count: int) -> bytes:
    output = bytearray()
    for current in packed:
        for shift in (6, 4, 2, 0):
            if len(output) == symbol_count:
                return bytes(output)
            output.append((current >> shift) & 0x03)
    if len(output) != symbol_count:
        raise ValueError("truncated 2-bit payload")
    return bytes(output)
