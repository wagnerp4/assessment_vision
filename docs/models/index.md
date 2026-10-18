# Models and backends

`ModelBackend` is the library-neutral adapter boundary. The built-in [OpenCV HSV backend](backends.md) needs no learned weights; an optional MediaPipe Holistic adapter provides person/hand landmarks. The core package remains importable without MediaPipe.

The [screw and nut detector](fastener-detector.md) is the first learned backend: an exported YOLO ONNX model run with ONNX Runtime or OpenCV DNN, with training tooling ready and weights not yet trained.

Other candidate PyTorch, Hugging Face Transformers, or other adapters are **planned**, not dependencies or currently supported backends. Add an adapter only after defining its normalized observations, license/weight distribution, device selection, startup/inference cost, deterministic tests, and evaluation. See [perception capabilities](../perception/index.md) for output types and [training/evaluation](../training-evaluation/index.md) for validation.
