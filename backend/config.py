import os
from threshold import THRESHOLD

# paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "resnet18_best.pth")

IMG_SIZE = 224
THRESHOLD = THRESHOLD

# normalization stats
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# server
PORT = int(os.environ.get("PORT", 5000))

# Set DEBUG=true in your environment for hot-reload during development.
# Never enable this in production.
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Upload validation

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}