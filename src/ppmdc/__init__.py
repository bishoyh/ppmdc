"""Prediction by partial matching DNA compressor."""

from .codec import FORMAT_VERSION, MAGIC, compress_bytes, compress_stream, decompress_stream

__all__ = ["FORMAT_VERSION", "MAGIC", "compress_bytes", "compress_stream", "decompress_stream"]
