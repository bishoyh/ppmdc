# ppmdc

`ppmdc` is a small experimental PPM compressor aimed at FASTA workloads.

The current implementation is an adaptive PPM compressor with FASTA-aware transforms, an experimental repeat-aware mode, a native C hot path, and block-based archives for larger inputs:

- It uses prediction by partial matching with context orders `0..16`.
- It preserves the original file exactly on decompression.
- It can auto-detect FASTA input and use a sampled prefix to choose between raw byte-wise coding and several FASTA-family transforms.
- It includes an explicit packed FASTA mode that stores 4-symbol sequence alphabets in a 2-bit representation.
- It includes an experimental repeat-aware mode that can emit backward and reverse-complement matches inside FASTA sequence groups.
- It splits file archives into independently compressed blocks so larger genomes can be processed with parallel workers.
- Its `ncd` command now defaults to a FASTA k-mer signature representation so related genomes share more structure before compression.
- It works well as a baseline for FASTA files because the alphabet and line structure are highly repetitive.

This is intentionally a simple research scaffold, not a production-grade archival format.

## Install

```bash
python3 -m pip install -e .
```

The editable install builds a native extension for the hot compression path. If native build is unavailable, the package falls back to the pure-Python codec, but it is much slower on large genomes.

## Usage

Compress a FASTA file:

```bash
ppmdc compress genome.fa --order 5
```

Use smaller blocks and more workers on bigger inputs:

```bash
ppmdc compress genome.fa --block-size-mb 8 --jobs 8
```

Force the byte-wise baseline instead of FASTA-aware mode:

```bash
ppmdc compress genome.fa --mode raw
```

Try the experimental repeat-aware FASTA mode:

```bash
ppmdc compress genome.fa --mode repeat
```

Try the 2-bit packed FASTA transform explicitly:

```bash
ppmdc compress genome.fa --mode packed
```

Decompress the resulting archive:

```bash
ppmdc decompress genome.fa.ppmdc
```

You can also run the module directly:

```bash
PYTHONPATH=src python3 -m ppmdc compress genome.fa
```

Compare `ppmdc` against `gzip` on one or more inputs:

```bash
ppmdc compare genome.fa another.fa
```

Compute a pairwise normalized compression distance matrix:

```bash
ppmdc ncd genome_a.fa genome_b.fa genome_c.fa
```

Use `gzip` as the backend for the NCD calculation instead of `ppmdc`:

```bash
ppmdc ncd genome_a.fa genome_b.fa genome_c.fa --backend gzip
```

Store persistent FASTA k-mer signatures in a chosen cache directory:

```bash
ppmdc ncd genome_a.fa genome_b.fa genome_c.fa --cache-dir /tmp/ppmdc-ncd-cache
```

Use the older whole-file NCD path instead of the FASTA k-mer signature:

```bash
ppmdc ncd genome_a.fa genome_b.fa genome_c.fa --representation file --mode auto
```

## Development

Run the test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Design Notes

- The compressor uses arithmetic coding for the final bitstream.
- The model is adaptive and updates after every byte.
- The PPM implementation uses escape symbols plus symbol exclusion when backing off to lower-order contexts.
- In FASTA mode, header lines and line endings are preserved exactly while sequence symbols are remapped to a smaller alphabet before coding.
- Packed mode uses the same exact FASTA metadata path, but stores 4-symbol sequence alphabets in a 2-bit payload.
- Repeat mode groups consecutive sequence lines, stores their line metadata separately, and tokenizes the concatenated sequence stream with literal runs, backward matches, and reverse-complement matches.
- The default NCD representation extracts canonical FASTA k-mers, applies deterministic subsampling, sorts the resulting set, caches per-genome signatures on disk, and merges those cached signatures for pairwise comparisons instead of rebuilding signatures from concatenated FASTA each time.
- The entropy-coding hot path, packed FASTA transform, and repeat-aware FASTA transform run in a C extension; Python remains responsible for archive handling and CLI behavior.
- Larger archives use a block format (`PPMDC`, version `3`, mode, order) with per-block metadata so blocks can be compressed independently.

## Limitations

- The scalable file path is block-streamed, but the in-memory `compress_bytes()` helper still operates on whole buffers.
- `auto` mode is a sampled heuristic for speed, not an exhaustive whole-file best-of search.
- Block boundaries reset model state, so block archives give up a small amount of compression ratio in exchange for parallelism and lower peak memory.
- Packed mode is exact, but it does not automatically improve compression; on the tested real genome it lost to the classic FASTA transform.
- The repeat-aware matcher is native now, but it remains experimental; on the tested real genomes it still did not beat the plain FASTA transform.
- NCD is still heuristic. The default k-mer signature is much more informative than whole-file compression for cross-genome comparison, but the result is not a phylogenetic model and still depends on `k`, subsampling, and the downstream compressor.
- Repeated `ncd` runs can now reuse cached per-genome signatures and cached compressed sizes for both single signatures and pairwise merged signatures, but cache size grows with the number of evaluated pairs.
- There is no block splitting, checksum, or random access support yet.

## Reference Benchmark

On the NCBI FASTA for `NC_000913.3` (*E. coli* K-12 MG1655, complete genome, 4,708,035 bytes), the current block-based implementation produced:

- `ppmdc_raw`: `1,194,677` bytes
- `ppmdc_fasta`: `1,193,542` bytes
- `ppmdc_packed`: `1,641,816` bytes
- `ppmdc_repeat`: `1,561,302` bytes
- `ppmdc_auto`: `1,193,542` bytes
- `gzip -9`: `1,387,836` bytes

On the same machine, `ppmdc compare data/bench/ecoli_k12_mg1655.fa` completed in about `9.2s` with `raw`, `fasta`, `packed`, `repeat`, and `gzip` all measured.

For a larger 37.7 MB stress test created by concatenating that real FASTA 8 times, the current `compare` output was:

- `ppmdc_raw`: `9,537,939` bytes
- `ppmdc_fasta`: `9,532,324` bytes
- `ppmdc_packed`: `13,119,594` bytes
- `ppmdc_repeat`: `12,462,868` bytes
- `ppmdc_auto`: `9,532,324` bytes
- `gzip -9`: `11,095,922` bytes

On the same machine:

- `ppmdc compare /tmp/ecoli_x8.fa`: about `31.5s`
- `ppmdc compress /tmp/ecoli_x8.fa --mode fasta --jobs 1`: about `3.40s`
- `ppmdc compress /tmp/ecoli_x8.fa --mode fasta --jobs 4`: about `1.35s`

The output size was identical for the `jobs=1` and `jobs=4` compression runs.

For a real NCD example on three bacterial reference genomes, the default signature-based command:

- `ppmdc ncd data/bench/ecoli_k12_mg1655.fa data/bench/salmonella_lt2.fa data/bench/bsubtilis_168.fa`

produced:

- `E. coli` K-12 MG1655 vs `Salmonella enterica` LT2: `0.754142`
- `E. coli` K-12 MG1655 vs `Bacillus subtilis` 168: `0.821896`
- `Salmonella enterica` LT2 vs `Bacillus subtilis` 168: `0.825861`

For comparison, the older whole-file path:

- `ppmdc ncd data/bench/ecoli_k12_mg1655.fa data/bench/salmonella_lt2.fa data/bench/bsubtilis_168.fa --representation file --mode auto`

gave:

- `E. coli` K-12 MG1655 vs `Salmonella enterica` LT2: `0.998603`
- `E. coli` K-12 MG1655 vs `Bacillus subtilis` 168: `1.001385`
- `Salmonella enterica` LT2 vs `Bacillus subtilis` 168: `1.002234`

The signature path is therefore the better default for similarity-oriented use.

With `--cache-dir /tmp/ppmdc-ncd-cache` on that same 3-genome example, the first run created `3` cached `.kms` signature files and `6` cached `.ksz` compressed-size entries. A cold run took about `13.0s`, and a repeated warm-cache run took about `0.08s`.
