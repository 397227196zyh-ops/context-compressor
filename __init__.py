"""context-compressor: Automatic context compression for LLM code agents."""
from context_compressor.core import compress_sync, create_compression_node
from context_compressor.types import CompressionConfig, CompressionResult

__version__ = "0.1.0"
__all__ = [
    "compress_sync", "create_compression_node",
    "CompressionConfig", "CompressionResult", "__version__",
]
