# Camera devices

The current app opens an integer OpenCV camera index. Windows selects DirectShow; other platforms use OpenCV's generic capture backend. Device numbering, supported capture APIs, permissions, resolution/FPS negotiation, and driver behavior are platform-specific. Report the selected index and negotiated properties; handle unavailable/open/read failures without crashing.

## WSL

WSL does not automatically expose a Windows webcam as a Linux `/dev/video*` device. WSL 2 can attach supported USB devices through Microsoft's `usbipd-win` USB/IP workflow; the USB device is unavailable to Windows while attached. Some integrated cameras may not be attachable this way. Check Linux device visibility before debugging application code. See [Microsoft's WSL USB guide](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

## Validation matrix (planned)

Record tested OS/version, OpenCV build/backend, camera model/connection, permissions, index, resolution/FPS, and whether concurrent Windows access is possible. Include open failure, unplug/reconnect, and read failure tests. Do not advertise an OS/device combination as supported based only on video-file processing.
