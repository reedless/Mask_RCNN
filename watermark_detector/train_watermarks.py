import os
import sys
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils, model as modellib

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

    # We use a GPU with 32GB memory
    IMAGES_PER_GPU = 4

    # Uncomment to train on 4 GPUs (default is 1)
    GPU_COUNT = 1

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

    # Problems w resizing when performing negative mining
    USE_MINI_MASK = False

    USE_RPN_ROIS = True

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

        # Get image ids
        image_ids = next(os.walk(dataset_dir))[2]
        print("Size of {} dataset: {}".format(subset, len(image_ids)))

        # Add images
        for image_id in image_ids:
            watermark_mask_file = os.path.join(root_dataset_dir, subset, 'mask_watermark', image_id)
            word_mask_file = os.path.join(root_dataset_dir, subset, 'mask_word', image_id)

            if os.path.isfile(watermark_mask_file) and os.path.isfile(word_mask_file):
                self.add_image(
                    "watermark",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id), 
                    mask_watermark=watermark_mask_file,
                    mask_word=word_mask_file
                )
            else:
                self.add_image(
                    "watermark",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id),
                    mask_watermark=None,
                    mask_word=None
                )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        # Get watermark and text masks from image info
        watermark_mask_file = info['mask_watermark']
        word_mask_file = info['mask_word']
        
        # if mask files do not exist, return empty masks and background id
        if watermark_mask_file is None and word_mask_file is None:
            img = cv2.imread(info['path'])
            img_width, img_height, _ = img.shape
            return np.ones([img_height, img_width, 1], dtype=np.uint8), np.zeros([1], dtype=np.int32)

        # Read mask files from disk
        watermark_mask_img = cv2.imread(watermark_mask_file, cv2.IMREAD_GRAYSCALE)
        word_mask_img = cv2.imread(word_mask_file, cv2.IMREAD_GRAYSCALE)
        
        img_width, img_height = watermark_mask_img.shape
        watermark_contours = self.get_contours(watermark_mask_img)
        word_contours = self.get_contours(word_mask_img)
        contours = watermark_contours + word_contours

        mask = np.zeros((img_width, img_height, len(contours)), dtype=np.uint8)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
        class_ids[len(watermark_contours):] = 2

        # convert contours to bitmask
        for i, contour in enumerate(contours):
            bitmask = np.zeros((img_width, img_height), dtype=np.uint8)
            contour_bitmask = cv2.drawContours(bitmask, [contour], -1, 1, -1)
            mask[:, :, i] = contour_bitmask

        return mask, class_ids

    def get_contours(self, mask):
        W = np.array(mask).astype(np.uint8)

        _, thresh = cv2.threshold(W, 100, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [contour for contour in contours if contour.shape[0] > 3]

        return contours


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "watermark":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = WatermarkDataset()
    dataset_train.load_watermark(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WatermarkDataset()
    dataset_val.load_watermark(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

# def test(model):
#     print("Testing implemented in predict_watermarks.py")
#     print(model)

# python train_watermark.py --weights=coco

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        default='/host/data/',
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the watermark dataset (default=data/')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=/logs/)')
    args = parser.parse_args()

    # # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "test":
    #     assert args.weights, "Provide --weights to test"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    # if args.command == "train":
    config = WatermarkConfig()
    # else:
    #     class InferenceConfig(WatermarkConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #     config = InferenceConfig()
    config.display()

    # Create model
    # if args.command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    # if args.command == "train":
    train(model)
    # elif args.command == "test":
    #     test(model)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'test'".format(args.command))