import os 
import skimage.draw
import sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.model import MaskRCNN
from mrcnn.config import Config

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Configuration
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "watermark"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + watermark + text

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None):
    assert image_path

    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        print(len(r['masks']))
        splash = color_splash(image, r['masks'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = "Splash.png"
        skimage.io.imsave(file_name, splash)
    print("Saved to", file_name)

# python predict_watermarks.py --image=/host/watermark_detector/Sample.jpg --weights=/host/logs/watermark20220104T0604/mask_rcnn_watermark_0030.h5
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect watermarks.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    assert args.image, "Provide --image to apply color splash"

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    config = InferenceConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Evaluate
    detect_and_color_splash(model, image_path=args.image)
