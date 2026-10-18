# User interfaces

Two interfaces exist:

- **Local window:** an OpenCV window with overlays on a camera or video frame, and keyboard controls. See [camera overlay](camera-overlay.md).
- **Browser client:** served by a small Python web server, deployed on Fly.io. See [web](web.md).

A [tablet](tablet.md) client is planned.

Keep presentation and input handling separate from model adapters. Interfaces should display task/session outputs and status, tolerate slow or unavailable inference, and make it clear when a displayed result is uncertain. End-of-game analysis differs per client: a summary window locally, and an analysis panel with charts and progression on the web (see [session analysis](../../analysis/session-analysis.md#4-presentation)).
