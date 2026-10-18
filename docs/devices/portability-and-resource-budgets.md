# Portability and resource budgets

**Status: compatibility goals; no formal laptop/resource support matrix yet.**

Keep the core contracts independent of OS and model library. Isolate capture backend selection, model loading, and device selection behind adapters. Video-file processing is the portable baseline; live camera capture depends on OS drivers and permissions. WSL requires explicit hardware passthrough for USB cameras.

For each supported tier, measure and document startup time, frame latency/throughput, CPU, peak RAM, GPU/VRAM, input resolution, and behavior when inference cannot keep pace. Define a CPU-only fallback before requiring GPU support. Avoid loading optional heavyweight frameworks unless their backend is selected. A low-resource mode may reduce input resolution, model size, or frame rate, but must report the resulting quality/temporal tradeoff and be evaluated for the target task.

The phrase “any laptop / any OS” is a goal, not a current guarantee; publish a tested matrix and known limitations as evidence accumulates.
