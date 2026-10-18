FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# MediaPipe's native library links GLES/EGL; its opencv-contrib dependency needs libGL/glib,
# and sounddevice needs PortAudio. Inference itself runs on the CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgles2 libegl1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy>=1.24.0" "scipy>=1.11.0" "mediapipe>=0.10.21,<0.11" \
    "fastapi>=0.110" "uvicorn[standard]>=0.29"

ADD https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task models/hand_landmarker.task
ADD https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task models/holistic_landmarker.task

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# Fail the build, not the first request, if a model cannot be loaded on Linux.
RUN python -c "from assessment_app.web import GameSession; [GameSession(mode).close() for mode in ('off', 'hands', 'holistic')]"

EXPOSE 8080
CMD ["uvicorn", "assessment_app.web:app", "--host", "0.0.0.0", "--port", "8080"]
