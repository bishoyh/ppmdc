from __future__ import annotations

import contextlib
from io import BytesIO, StringIO
from pathlib import Path
import random
import tempfile
import unittest

from ppmdc.cli import main
from ppmdc.codec import DEFAULT_ORDER, compress_bytes, compress_file, compress_stream, decompress_file, decompress_stream
from ppmdc.fasta import SEQUENCE_GROUP_PACKED_TAG, SEQUENCE_GROUP_TAG, SEQUENCE_PACKED_TAG, build_kmer_signature, merge_kmer_signatures, transform_fasta_packed, transform_fasta_repeats


class CodecTests(unittest.TestCase):
    def test_roundtrip_preserves_bytes(self) -> None:
        payload = (
            b">chr1 example\n"
            b"ACGTACGTNNNNACGTACGT\n"
            b">chr2\n"
            b"TTTTAAAACCCCGGGG\n"
        )

        compressed = BytesIO()
        compress_stream(BytesIO(payload), compressed, order=DEFAULT_ORDER)
        restored = BytesIO()
        decompress_stream(BytesIO(compressed.getvalue()), restored)

        self.assertEqual(restored.getvalue(), payload)

    def test_repetitive_fasta_compresses(self) -> None:
        sequence = (b"ACGT" * 4096) + b"\n"
        payload = b">synthetic\n" + sequence * 8

        compressed = BytesIO()
        compress_stream(BytesIO(payload), compressed, order=DEFAULT_ORDER)

        self.assertLess(len(compressed.getvalue()), len(payload))

    def test_invalid_header_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decompress_stream(BytesIO(b"not-a-ppmdc-file"), BytesIO())

    def test_fasta_mode_roundtrip_preserves_headers_and_crlf(self) -> None:
        payload = (
            b">chr1 description\r\n"
            b"ACGTNacgtn\r\n"
            b">chr2\r\n"
            b"TTTTAAAA\r\n"
        )

        result = compress_bytes(payload, order=DEFAULT_ORDER, mode="fasta")
        restored = BytesIO()
        written = decompress_stream(BytesIO(result.archive), restored)

        self.assertEqual(result.mode, "fasta")
        self.assertEqual(restored.getvalue(), payload)
        self.assertEqual(written, len(payload))

    def test_auto_mode_selects_smaller_fasta_archive_when_beneficial(self) -> None:
        rng = random.Random(7)
        sequence = bytes(rng.choice(b"ACGT") for _ in range(120000))
        lines = [b">synthetic\n"]
        for index in range(0, len(sequence), 80):
            lines.append(sequence[index : index + 80] + b"\n")
        payload = b"".join(lines)

        raw_result = compress_bytes(payload, order=DEFAULT_ORDER, mode="raw")
        fasta_result = compress_bytes(payload, order=DEFAULT_ORDER, mode="fasta")
        repeat_result = compress_bytes(payload, order=DEFAULT_ORDER, mode="repeat")
        auto_result = compress_bytes(payload, order=DEFAULT_ORDER, mode="auto")

        sizes = {
            "raw": len(raw_result.archive),
            "fasta": len(fasta_result.archive),
            "repeat": len(repeat_result.archive),
        }
        self.assertEqual(auto_result.mode, min(sizes, key=sizes.__getitem__))

    def test_repeat_mode_roundtrip_preserves_reverse_complements(self) -> None:
        seed = (b"ACGTTGCA" * 10)
        reverse_complement = bytes({ord("A"): ord("T"), ord("C"): ord("G"), ord("G"): ord("C"), ord("T"): ord("A")}[base] for base in reversed(seed))
        payload = b">chr1\n" + seed + b"\n>chr2\n" + reverse_complement + b"\n"

        result = compress_bytes(payload, order=DEFAULT_ORDER, mode="repeat")
        restored = BytesIO()
        decompress_stream(BytesIO(result.archive), restored)

        self.assertEqual(result.mode, "repeat")
        self.assertEqual(restored.getvalue(), payload)

    def test_repeat_transform_emits_repeat_group_when_packing_is_unavailable(self) -> None:
        motif = b"ACGTNGATTACAACCGT" * 6
        payload, alphabet = transform_fasta_repeats(
            b">chr1\n" + motif + b"\n>chr2\n" + motif + b"\n>chr3\n" + motif + b"\n"
        )

        self.assertEqual(alphabet, b"ACGNT")
        self.assertIn(SEQUENCE_GROUP_TAG, payload)

    def test_compare_command_reports_gzip_and_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fasta_path = Path(tmp) / "sample.fa"
            fasta_path.write_bytes(b">sample\n" + (b"ACGT\n" * 64))

            output = StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["compare", str(fasta_path)])

            self.assertEqual(exit_code, 0)
            rendered = output.getvalue()
            self.assertIn("ppmdc_auto", rendered)
            self.assertIn("ppmdc_fasta", rendered)
            self.assertIn("ppmdc_packed", rendered)
            self.assertIn("ppmdc_repeat", rendered)
            self.assertIn("gzip", rendered)
            self.assertIn("selected_mode", rendered)

    def test_transform_fasta_uses_two_bit_packing_for_four_symbol_alphabet(self) -> None:
        source = b">sample\nACGTACGTACGT\n"
        payload, alphabet = transform_fasta_packed(source)
        self.assertEqual(alphabet, b"ACGT")
        first_sequence_tag = 1 + 1 + len(b"sample") + 1
        self.assertEqual(payload[first_sequence_tag], SEQUENCE_PACKED_TAG)

    def test_repeat_transform_can_use_group_packed_tag(self) -> None:
        source = b">sample\nACGTACGT\nTGCATGCA\nCATGCATG\n"
        payload, alphabet = transform_fasta_repeats(source)
        self.assertEqual(alphabet, b"ACGT")
        first_sequence_tag = 1 + 1 + len(b"sample") + 1
        self.assertEqual(payload[first_sequence_tag], SEQUENCE_GROUP_PACKED_TAG)

    def test_kmer_signature_is_line_and_strand_invariant(self) -> None:
        sequence = b"ACGTGCAATTCGACGTGCAA"
        complement = {ord("A"): ord("T"), ord("C"): ord("G"), ord("G"): ord("C"), ord("T"): ord("A")}
        reverse_complement = bytes(complement[base] for base in reversed(sequence))

        single_line = b">a\n" + sequence + b"\n"
        wrapped = b">a\n" + sequence[:10] + b"\n" + sequence[10:] + b"\n"
        opposite_strand = b">b\n" + reverse_complement[:8] + b"\n" + reverse_complement[8:] + b"\n"

        reference = build_kmer_signature(single_line, kmer_size=7, sample_rate=1)
        self.assertEqual(reference, build_kmer_signature(wrapped, kmer_size=7, sample_rate=1))
        self.assertEqual(reference, build_kmer_signature(opposite_strand, kmer_size=7, sample_rate=1))

    def test_merge_kmer_signatures_matches_combined_fasta(self) -> None:
        left = b">a\nACGTACGTACGT\n"
        right = b">b\nTTTTCCCCAAAA\n"
        merged = merge_kmer_signatures(
            build_kmer_signature(left, kmer_size=5, sample_rate=1),
            build_kmer_signature(right, kmer_size=5, sample_rate=1),
        )
        combined = build_kmer_signature(left + right, kmer_size=5, sample_rate=1)
        self.assertEqual(merged, combined)

    def test_ncd_command_ranks_similar_genomes_closer(self) -> None:
        rng = random.Random(11)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genome_a = root / "a.fa"
            genome_b = root / "b.fa"
            genome_c = root / "c.fa"

            base = bytearray(rng.choice(b"ACGT") for _ in range(16000))
            similar = bytearray(base)
            for index in range(0, len(similar), 400):
                similar[index] = ord("A") if similar[index] != ord("A") else ord("C")
            different = bytearray(rng.choice(b"ACGT") for _ in range(16000))

            genome_a.write_bytes(b">a\n" + bytes(base) + b"\n")
            genome_b.write_bytes(b">b\n" + bytes(similar) + b"\n")
            genome_c.write_bytes(b">c\n" + bytes(different) + b"\n")

            output = StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "ncd",
                        str(genome_a),
                        str(genome_b),
                        str(genome_c),
                        "--representation",
                        "kmer",
                        "--kmer-size",
                        "11",
                        "--sample-rate",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            lines = output.getvalue().strip().splitlines()
            header = lines[0].split("\t")[1:]
            matrix: dict[str, dict[str, float]] = {}
            for line in lines[1:]:
                cells = line.split("\t")
                matrix[cells[0]] = {
                    column: float(value)
                    for column, value in zip(header, cells[1:])
                }

            self.assertAlmostEqual(matrix[str(genome_a)][str(genome_a)], 0.0)
            self.assertLess(matrix[str(genome_a)][str(genome_b)], matrix[str(genome_a)][str(genome_c)])

    def test_ncd_kmer_cache_dir_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            genome_a = root / "a.fa"
            genome_b = root / "b.fa"

            genome_a.write_bytes(b">a\n" + (b"ACGT" * 128) + b"\n")
            genome_b.write_bytes(b">b\n" + (b"ACGA" * 128) + b"\n")

            output = StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "ncd",
                        str(genome_a),
                        str(genome_b),
                        "--representation",
                        "kmer",
                        "--cache-dir",
                        str(cache_dir),
                        "--kmer-size",
                        "7",
                        "--sample-rate",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(any(cache_dir.glob("*.kms")))
            self.assertTrue(any(cache_dir.glob("*.ksz")))

    def test_file_roundtrip_uses_block_archive_and_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.fa"
            archive = root / "sample.fa.ppmdc"
            restored = root / "sample.fa.restored"
            payload = b">sample\n" + (b"ACGTN" * 2000) + b"\n"
            source.write_bytes(payload)

            compress_result = compress_file(source, archive, mode="auto", block_size=64, jobs=2)
            decompress_result = decompress_file(archive, restored)

            self.assertEqual(compress_result.input_size, len(payload))
            self.assertEqual(decompress_result.output_size, len(payload))
            self.assertEqual(restored.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
