"""Command-line interface for ppmdc."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .codec import DEFAULT_BLOCK_SIZE, DEFAULT_ORDER, compress_bytes, compress_file, decompress_file
from .fasta import build_kmer_signature, merge_kmer_signatures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppmdc",
        description="Prediction by partial matching compressor for FASTA and other byte streams.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="compress a file")
    compress_parser.add_argument("input", type=Path, help="input file to compress")
    compress_parser.add_argument("-o", "--output", type=Path, help="output archive path")
    compress_parser.add_argument(
        "--order",
        type=int,
        default=DEFAULT_ORDER,
        help=f"maximum context order (default: {DEFAULT_ORDER})",
    )
    compress_parser.add_argument(
        "--mode",
        choices=("auto", "raw", "fasta", "packed", "repeat"),
        default="auto",
        help="compression mode: sample-select among raw and FASTA-family transforms for FASTA-like input, raw bytes, or an explicit transform",
    )
    compress_parser.add_argument(
        "--block-size-mb",
        type=int,
        default=DEFAULT_BLOCK_SIZE // (1024 * 1024),
        help="uncompressed block size in MiB for file compression (default: 4)",
    )
    compress_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel worker processes for block compression (default: auto)",
    )
    compress_parser.set_defaults(handler=_handle_compress)

    decompress_parser = subparsers.add_parser("decompress", help="decompress an archive")
    decompress_parser.add_argument("input", type=Path, help="archive to decompress")
    decompress_parser.add_argument("-o", "--output", type=Path, help="output file path")
    decompress_parser.set_defaults(handler=_handle_decompress)

    compare_parser = subparsers.add_parser("compare", help="compare ppmdc to gzip")
    compare_parser.add_argument("inputs", nargs="+", type=Path, help="input files to benchmark")
    compare_parser.add_argument(
        "--order",
        type=int,
        default=DEFAULT_ORDER,
        help=f"maximum context order for ppmdc (default: {DEFAULT_ORDER})",
    )
    compare_parser.add_argument(
        "--gzip-level",
        type=int,
        default=9,
        choices=range(1, 10),
        metavar="{1..9}",
        help="gzip compression level (default: 9)",
    )
    compare_parser.add_argument(
        "--block-size-mb",
        type=int,
        default=DEFAULT_BLOCK_SIZE // (1024 * 1024),
        help="uncompressed block size in MiB for ppmdc file compression (default: 4)",
    )
    compare_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel worker processes for ppmdc block compression (default: auto)",
    )
    compare_parser.set_defaults(handler=_handle_compare)

    ncd_parser = subparsers.add_parser("ncd", help="compute normalized compression distance between files")
    ncd_parser.add_argument("inputs", nargs="+", type=Path, help="input files to compare")
    ncd_parser.add_argument(
        "--backend",
        choices=("ppmdc", "gzip"),
        default="ppmdc",
        help="compressor backend used to compute NCD (default: ppmdc)",
    )
    ncd_parser.add_argument(
        "--representation",
        choices=("file", "kmer"),
        default="kmer",
        help="compare raw files or FASTA k-mer signatures before compression (default: kmer)",
    )
    ncd_parser.add_argument(
        "--mode",
        choices=("auto", "raw", "fasta", "packed", "repeat"),
        default="auto",
        help="ppmdc compression mode for NCD when backend=ppmdc and representation=file (default: auto)",
    )
    ncd_parser.add_argument(
        "--order",
        type=int,
        default=DEFAULT_ORDER,
        help=f"maximum context order for ppmdc NCD (default: {DEFAULT_ORDER})",
    )
    ncd_parser.add_argument(
        "--gzip-level",
        type=int,
        default=9,
        choices=range(1, 10),
        metavar="{1..9}",
        help="gzip compression level for NCD when backend=gzip (default: 9)",
    )
    ncd_parser.add_argument(
        "--kmer-size",
        type=int,
        default=15,
        help="canonical k-mer size for representation=kmer (default: 15)",
    )
    ncd_parser.add_argument(
        "--sample-rate",
        type=int,
        default=8,
        help="keep about 1/N canonical k-mers in representation=kmer signatures (default: 8)",
    )
    ncd_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="directory for persistent FASTA k-mer signature cache when representation=kmer",
    )
    ncd_parser.add_argument(
        "--block-size-mb",
        type=int,
        default=DEFAULT_BLOCK_SIZE // (1024 * 1024),
        help="uncompressed block size in MiB for ppmdc NCD (default: 4)",
    )
    ncd_parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel worker processes for ppmdc NCD block compression (default: auto)",
    )
    ncd_parser.set_defaults(handler=_handle_ncd)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _handle_compress(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    output_path = args.output or input_path.with_suffix(input_path.suffix + ".ppmdc")
    result = compress_file(
        input_path,
        output_path,
        order=args.order,
        mode=args.mode,
        block_size=_block_size_bytes(args.block_size_mb),
        jobs=args.jobs,
    )
    ratio = result.output_size / result.input_size if result.input_size else 0.0
    print(
        f"compressed {input_path} -> {output_path} "
        f"using {result.mode} mode "
        f"({result.input_size} bytes -> {result.output_size} bytes, ratio {ratio:.3f})"
    )
    return 0


def _handle_decompress(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    output_path = args.output or _default_output_path(input_path)
    result = decompress_file(input_path, output_path)
    ratio = result.output_size / result.input_size if result.input_size else 0.0
    print(
        f"decompressed {input_path} -> {output_path} "
        f"({result.input_size} bytes -> {result.output_size} bytes, ratio {ratio:.3f})"
    )
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    print("file\tinput\tppmdc_raw\tppmdc_fasta\tppmdc_packed\tppmdc_repeat\tppmdc_auto\tselected_mode\tgzip")

    block_size = _block_size_bytes(args.block_size_mb)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for input_path in args.inputs:
            input_size = input_path.stat().st_size
            raw_result = compress_file(
                input_path,
                root / f"{input_path.name}.raw.ppmdc",
                order=args.order,
                mode="raw",
                block_size=block_size,
                jobs=args.jobs,
            )
            auto_result = compress_file(
                input_path,
                root / f"{input_path.name}.auto.ppmdc",
                order=args.order,
                mode="auto",
                block_size=block_size,
                jobs=args.jobs,
            )
            try:
                fasta_size = str(
                    compress_file(
                        input_path,
                        root / f"{input_path.name}.fasta.ppmdc",
                        order=args.order,
                        mode="fasta",
                        block_size=block_size,
                        jobs=args.jobs,
                    ).output_size
                )
            except ValueError:
                fasta_size = "NA"

            try:
                packed_size = str(
                    compress_file(
                        input_path,
                        root / f"{input_path.name}.packed.ppmdc",
                        order=args.order,
                        mode="packed",
                        block_size=block_size,
                        jobs=args.jobs,
                    ).output_size
                )
            except ValueError:
                packed_size = "NA"

            try:
                repeat_size = str(
                    compress_file(
                        input_path,
                        root / f"{input_path.name}.repeat.ppmdc",
                        order=args.order,
                        mode="repeat",
                        block_size=block_size,
                        jobs=args.jobs,
                    ).output_size
                )
            except ValueError:
                repeat_size = "NA"

            gzip_archive = gzip.compress(input_path.read_bytes(), compresslevel=args.gzip_level, mtime=0)
            print(
                f"{input_path}\t{input_size}\t{raw_result.output_size}\t{fasta_size}\t{packed_size}\t{repeat_size}\t"
                f"{auto_result.output_size}\t{auto_result.mode}\t{len(gzip_archive)}"
            )

    return 0


def _handle_ncd(args: argparse.Namespace) -> int:
    if len(args.inputs) < 2:
        raise SystemExit("ncd requires at least two input files")

    block_size = _block_size_bytes(args.block_size_mb)
    if args.representation == "kmer":
        _handle_ncd_kmer(args, block_size)
        return 0

    cache: dict[tuple[str, ...], int] = {}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_sizes = {
            path: _compressed_size(
                (path,),
                root,
                cache,
                backend=args.backend,
                representation=args.representation,
                mode=args.mode,
                order=args.order,
                gzip_level=args.gzip_level,
                kmer_size=args.kmer_size,
                sample_rate=args.sample_rate,
                block_size=block_size,
                jobs=args.jobs,
            )
            for path in args.inputs
        }

        matrix: dict[tuple[Path, Path], float] = {}
        for row_index, left in enumerate(args.inputs):
            for col_index, right in enumerate(args.inputs):
                if row_index == col_index:
                    matrix[(left, right)] = 0.0
                    continue
                if col_index < row_index:
                    matrix[(left, right)] = matrix[(right, left)]
                    continue

                combined_size = _compressed_size(
                    (left, right),
                    root,
                    cache,
                    backend=args.backend,
                    representation=args.representation,
                    mode=args.mode,
                    order=args.order,
                    gzip_level=args.gzip_level,
                    kmer_size=args.kmer_size,
                    sample_rate=args.sample_rate,
                    block_size=block_size,
                    jobs=args.jobs,
                )
                matrix[(left, right)] = (
                    combined_size - min(base_sizes[left], base_sizes[right])
                ) / max(base_sizes[left], base_sizes[right])

        print("file\t" + "\t".join(str(path) for path in args.inputs))
        for left in args.inputs:
            print(
                str(left)
                + "\t"
                + "\t".join(f"{matrix[(left, right)]:.6f}" for right in args.inputs)
            )

    return 0


def _handle_ncd_kmer(args: argparse.Namespace, block_size: int) -> None:
    cache_dir = args.cache_dir or _default_ncd_cache_dir()
    signature_cache: dict[tuple[str, int, int, int, int, int, int], tuple[bytes, str]] = {}
    signatures = {
        path: _load_or_build_kmer_signature(
            path,
            cache_dir,
            kmer_size=args.kmer_size,
            sample_rate=args.sample_rate,
            memory_cache=signature_cache,
        )
        for path in args.inputs
    }

    size_cache: dict[tuple[str, ...], int] = {}
    base_sizes = {
        path: _compressed_kmer_size(
            signatures[path][0],
            cache_dir,
            size_cache,
            key=_kmer_size_cache_key(
                "single",
                args.backend,
                args.order,
                args.gzip_level,
                block_size,
                args.jobs,
                signatures[path][1],
            ),
            backend=args.backend,
            order=args.order,
            gzip_level=args.gzip_level,
            block_size=block_size,
            jobs=args.jobs,
        )
        for path in args.inputs
    }

    matrix: dict[tuple[Path, Path], float] = {}
    for row_index, left in enumerate(args.inputs):
        for col_index, right in enumerate(args.inputs):
            if row_index == col_index:
                matrix[(left, right)] = 0.0
                continue
            if col_index < row_index:
                matrix[(left, right)] = matrix[(right, left)]
                continue

            pair_ids = tuple(sorted((signatures[left][1], signatures[right][1])))
            combined_key = _kmer_size_cache_key(
                "pair",
                args.backend,
                args.order,
                args.gzip_level,
                block_size,
                args.jobs,
                *pair_ids,
            )
            combined_size = _load_cached_size(cache_dir, size_cache, combined_key)
            if combined_size is None:
                merged_signature = merge_kmer_signatures(signatures[left][0], signatures[right][0])
                combined_size = _compressed_kmer_size(
                    merged_signature,
                    cache_dir,
                    size_cache,
                    key=combined_key,
                    backend=args.backend,
                    order=args.order,
                    gzip_level=args.gzip_level,
                    block_size=block_size,
                    jobs=args.jobs,
                )
            matrix[(left, right)] = (
                combined_size - min(base_sizes[left], base_sizes[right])
            ) / max(base_sizes[left], base_sizes[right])

    print("file\t" + "\t".join(str(path) for path in args.inputs))
    for left in args.inputs:
        print(
            str(left)
            + "\t"
            + "\t".join(f"{matrix[(left, right)]:.6f}" for right in args.inputs)
        )


def _compressed_size(
    inputs: tuple[Path, ...],
    root: Path,
    cache: dict[tuple[str, ...], int],
    *,
    backend: str,
    representation: str,
    mode: str,
    order: int,
    gzip_level: int,
    kmer_size: int,
    sample_rate: int,
    block_size: int,
    jobs: int,
) -> int:
    key = (
        backend,
        representation,
        mode,
        str(order),
        str(gzip_level),
        str(kmer_size),
        str(sample_rate),
        str(block_size),
        str(jobs),
        *[str(path) for path in inputs],
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha1("\0".join(key).encode("utf-8")).hexdigest()
    input_path = root / f"{digest}.input"
    _concatenate_inputs(
        input_path,
        inputs,
        ensure_fasta_boundary=representation == "kmer" or (backend == "ppmdc" and mode != "raw"),
    )

    compression_input_path = input_path
    compression_mode = mode
    if representation == "kmer":
        signature_path = root / f"{digest}.kmer"
        signature_path.write_bytes(
            build_kmer_signature(
                input_path.read_bytes(),
                kmer_size=kmer_size,
                sample_rate=sample_rate,
            )
        )
        compression_input_path = signature_path
        compression_mode = "raw"

    if backend == "ppmdc":
        archive_path = root / f"{digest}.ppmdc"
        size = compress_file(
            compression_input_path,
            archive_path,
            order=order,
            mode=compression_mode,
            block_size=block_size,
            jobs=jobs,
        ).output_size
    else:
        archive_path = root / f"{digest}.gz"
        size = _gzip_file_size(compression_input_path, archive_path, gzip_level)

    cache[key] = size
    return size


def _load_or_build_kmer_signature(
    input_path: Path,
    cache_dir: Path,
    *,
    kmer_size: int,
    sample_rate: int,
    memory_cache: dict[tuple[str, int, int, int, int, int, int], tuple[bytes, str]],
) -> tuple[bytes, str]:
    stat = input_path.stat()
    cache_key = (
        str(input_path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        getattr(stat, "st_ino", 0),
        getattr(stat, "st_dev", 0),
        kmer_size,
        sample_rate,
    )
    cached = memory_cache.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha1(
        "\0".join(
            (
                str(input_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(getattr(stat, "st_ino", 0)),
                str(getattr(stat, "st_dev", 0)),
                str(kmer_size),
                str(sample_rate),
            )
        ).encode("utf-8")
    ).hexdigest()
    signature_path = cache_dir / f"{digest}.kms"
    if signature_path.exists():
        signature = signature_path.read_bytes()
        entry = (signature, digest)
        memory_cache[cache_key] = entry
        return entry

    signature = build_kmer_signature(
        input_path.read_bytes(),
        kmer_size=kmer_size,
        sample_rate=sample_rate,
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="sig-", suffix=".tmp", delete=False) as temporary:
            temporary.write(signature)
            temporary_path = Path(temporary.name)
        temporary_path.replace(signature_path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass

    entry = (signature, digest)
    memory_cache[cache_key] = entry
    return entry


def _compressed_kmer_size(
    data: bytes,
    cache_dir: Path,
    cache: dict[tuple[str, ...], int],
    *,
    key: tuple[str, ...],
    backend: str,
    order: int,
    gzip_level: int,
    block_size: int,
    jobs: int,
) -> int:
    cached = _load_cached_size(cache_dir, cache, key)
    if cached is not None:
        return cached

    size = _compute_compressed_buffer_size(
        data,
        backend=backend,
        order=order,
        gzip_level=gzip_level,
        block_size=block_size,
        jobs=jobs,
    )
    _store_cached_size(cache_dir, cache, key, size)
    return size


def _concatenate_inputs(output_path: Path, inputs: tuple[Path, ...], *, ensure_fasta_boundary: bool) -> None:
    with output_path.open("wb") as target:
        for index, input_path in enumerate(inputs):
            last_byte = _copy_file(target, input_path)
            if ensure_fasta_boundary and index + 1 < len(inputs) and last_byte not in (None, 0x0A, 0x0D):
                target.write(b"\n")


def _copy_file(target, input_path: Path) -> int | None:
    last_byte: int | None = None
    with input_path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return last_byte
            target.write(chunk)
            last_byte = chunk[-1]


def _gzip_file_size(input_path: Path, archive_path: Path, level: int) -> int:
    with input_path.open("rb") as source, archive_path.open("wb") as raw_target:
        with gzip.GzipFile(fileobj=raw_target, mode="wb", compresslevel=level, mtime=0) as gz_target:
            shutil.copyfileobj(source, gz_target, length=1024 * 1024)
    return archive_path.stat().st_size


def _kmer_size_cache_key(
    kind: str,
    backend: str,
    order: int,
    gzip_level: int,
    block_size: int,
    jobs: int,
    *signature_ids: str,
) -> tuple[str, ...]:
    return (
        kind,
        backend,
        str(order),
        str(gzip_level),
        str(block_size),
        str(jobs),
        *signature_ids,
    )


def _size_cache_path(cache_dir: Path, key: tuple[str, ...]) -> Path:
    digest = hashlib.sha1("\0".join(key).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.ksz"


def _load_cached_size(cache_dir: Path, cache: dict[tuple[str, ...], int], key: tuple[str, ...]) -> int | None:
    cached = cache.get(key)
    if cached is not None:
        return cached

    size_path = _size_cache_path(cache_dir, key)
    try:
        raw = size_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        size = int(raw)
    except ValueError:
        return None

    cache[key] = size
    return size


def _store_cached_size(cache_dir: Path, cache: dict[tuple[str, ...], int], key: tuple[str, ...], size: int) -> None:
    cache[key] = size
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="size-", suffix=".tmp", delete=False) as temporary:
            temporary.write(f"{size}\n".encode("ascii"))
            temporary_path = Path(temporary.name)
        temporary_path.replace(_size_cache_path(cache_dir, key))
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass


def _compute_compressed_buffer_size(
    data: bytes,
    *,
    backend: str,
    order: int,
    gzip_level: int,
    block_size: int,
    jobs: int,
) -> int:
    if backend == "ppmdc":
        return len(
            compress_bytes(
                data,
                order=order,
                mode="raw",
                block_size=block_size,
                jobs=jobs,
            ).archive
        )
    return len(gzip.compress(data, compresslevel=gzip_level, mtime=0))


def _default_ncd_cache_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "ppmdc" / "ncd"
    return Path.home() / ".cache" / "ppmdc" / "ncd"


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".ppmdc":
        return input_path.with_suffix("")
    return input_path.with_suffix(input_path.suffix + ".out")


def _block_size_bytes(block_size_mb: int) -> int:
    if block_size_mb <= 0:
        raise SystemExit("--block-size-mb must be positive")
    return block_size_mb * 1024 * 1024


if __name__ == "__main__":
    raise SystemExit(main())
