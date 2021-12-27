import os
import sys
import numpy as np
import cv2

ROOT_DIR = os.path.abspath("../")

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

class WatermarkDataset(utils.Dataset):

    def load_watermark(self, root_dataset_dir, subset):
        """Load a subset of the watermark dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have two classes to add.
        self.add_class("watermark", 1, "watermark")
        self.add_class("watermark", 2, "text")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(root_dataset_dir, subset, 'input')
        mask_watermark_dir = os.path.join(root_dataset_dir, subset, 'mask_watermark')
        mask_word_dir = os.path.join(root_dataset_dir, subset, 'mask_word')
        print(dataset_dir, mask_watermark_dir, mask_word_dir)

        # Get image ids from directory
        image_ids = next(os.walk(dataset_dir))[2]
        print("Size of {} dataset: {}".format(subset, len(image_ids)))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "watermark",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id), 
                mask_watermark=os.path.join(mask_watermark_dir, image_id),
                mask_word=os.path.join(mask_word_dir, image_id))

    # TODO: add text masks, edit functions to add masks
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'mask_watermark')
        # Read mask files from .jpg image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".jpg"):
                m = cv2.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "watermark":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)










































def main():
    print("Hello World!")

if __name__ == '__main__':
    main()