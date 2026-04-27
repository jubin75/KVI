"""
Package: model_adapters

Responsibilities:
- Place adapters for different model families (Qwen first, DeepSeek pluggable).
- Expose a unified ModelAdapter abstraction and factory method semantics to the upper layers.

Constraints:
- Adapters must ensure: controllable injection switch, consistent KV cache behavior,
  dimensional consistency, and observable debug output.
"""


