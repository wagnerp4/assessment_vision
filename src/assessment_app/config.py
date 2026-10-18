from dataclasses import dataclass


@dataclass
class CubeConfig:
    """Configuration for cube detection and game logic"""

    # Cube size constraints (in cm)
    MAX_CUBE_WIDTH_CM = 3.0

    # Color definitions (BGR format for OpenCV)
    CUBE_COLORS = {
        "blue": ((88, 73, 51), (118, 133, 111)),
        "green": ((42, 36, 55), (72, 96, 115)),
        "red": ((100, 28, 156), (130, 88, 216)),
        "yellow": ((8, 60, 173), (38, 120, 233)),
    }

    # Detection thresholds (CALIBRATED FOR YOUR CUBES)
    CAMERA_EXPOSURE = -6  # Manual exposure (DirectShow log2 seconds); None = camera auto-exposure
    MIN_CONTOUR_AREA = (
        1200  # Minimum pixels for a valid cube; size-relative filtering removes clutter
    )
    MAX_CONTOUR_AREA = 150000  # Maximum pixels for a valid cube (was 100000)
    COLOR_MATCH_THRESHOLD = 0.5  # Color matching confidence (0-1) (lowered)

    # Alignment detection
    EDGE_ALIGNMENT_TOLERANCE_PX = 10  # pixels tolerance for edge alignment
    EDGE_ALIGNMENT_TOLERANCE_PERCENT = 0.05  # 5% tolerance

    # Stacking detection
    VERTICAL_OVERLAP_THRESHOLD = 0.7  # How much top cube overlaps bottom (0-1)
    VERTICAL_DISTANCE_THRESHOLD = 30  # Max pixels gap between cubes vertically

    # Frame processing
    FRAME_SKIP_RATE = 2  # Process every Nth frame
    FRAME_RESIZE_HEIGHT = 720  # Resize frames for processing

    # Tracking
    TRACKING_HISTORY_SIZE = 30  # Frames to keep history
    CUBE_ID_TIMEOUT = 60  # Frames before losing track of cube

    # Game logic
    VALID_STACK_SEQUENCES = [
        ("blue", "red", "green", "yellow"),
        ("yellow", "green", "red", "blue"),
        ("red", "blue", "yellow", "green"),
    ]

    STACK_CONFIRMATION_FRAMES = 10  # Frames to confirm stable stacking
