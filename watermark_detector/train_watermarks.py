import os
import sys

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class WatermarkConfig(Config):
    """Configuration for training on the watermark dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "watermark"

    # We use a GPU with 32GB memory, which can fit 180(?) images
    IMAGES_PER_GPU = 180

    # Uncomment to train on 4 GPUs (default is 1)
    # GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # watermark and text

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

############################################################
#  Dataset
############################################################










































def main():
    print("Hello World!")

if __name__ == '__main__':
    main()