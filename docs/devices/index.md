# Devices and runtime portability

The application currently uses OpenCV capture and can run on Windows and Linux environments with compatible Python/OpenCV builds. This does not guarantee that every laptop, OS, camera, or accelerator is supported. WSL adds a hardware-passthrough boundary; camera devices are not automatically visible inside the Linux guest.

Document and test the actual supported matrix rather than promising universal hardware support. See [cameras](cameras.md) and [portability/resource budgets](portability-and-resource-budgets.md).
