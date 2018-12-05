"""
Mask R-CNN
Train on the toy lanes dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 lanes.py train --dataset=/path/to/lanes/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 lanes.py train --dataset=/path/to/lanes/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 lanes.py train --dataset=/path/to/lanes/dataset --weights=imagenet
    # Apply color splash to an image
    python3 lanes.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 lanes.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
import cv2
import time

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/lanes"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from config import Config
import utils
import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class lanesConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lanes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128

    IMAGE_SHAPE = np.array((720, 1280, 3))

    # IMAGE_WIDTH = 1280
    # IMAGE_HEIGHT = 720
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024) # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32


    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    LEARNING_RATE = 0.001

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128) 

############################################################
#  Dataset
############################################################

class lanesDataset(utils.Dataset):

    def load_lanes(self, subset):
        """Load a subset of the lanes dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("lanes", 1, "lanes")

        fimgs = glob.glob('/media/ml3/Volume/sg/sentosa_lane_label/image/*.png')
        
        if subset == "train":
            k = 0 
            m = int(0.9*len(fimgs))
        else:
            k = int(0.9*len(fimgs))
            m = len(fimgs)


        print(m)
        # Add images
        for i in range(k, m):

            image_path = fimgs[i]
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "lanes",
                image_id=i,  # use file name as a unique image id
                path=image_path,
                width=width, height=height
                )
        

            # break


        
        # color = [[230, 0, 230], [0, 255, 255], [255, 255, 0], [0, 0, 255]]

        # for i in range(len(self.image_info)):    
        #     mask, label = self.load_mask(i)
        #     print(label)
        #     image_path = self.image_info[i]["path"]

        #     img = cv2.imread(self.image_reference(i))

        #     res = np.zeros_like(img)
                    
        #     for i in range(mask.shape[2]):
        #         ind = mask[:, :, i] == 1

        #         res[ind] = color[i%len(color)]

        #     res = cv2.addWeighted(img, 1, res, 1, 0)
        #     cv2.imshow("res", res)
        #     cv2.waitKey(0)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # print(len(self.image_info), self.image_info)        
        info = self.image_info[image_id]

        img_name = info['path'].split('/')[-1]
        init_mask = cv2.imread('/media/ml3/Volume/sg/sentosa_lane_label/gt_image_instance/'+img_name)[:, :, 0]
        unique = np.unique(init_mask)
        # print(init_mask.shape)

        mask = np.zeros((init_mask.shape[0], init_mask.shape[1], len(unique)-1))
        
        for i in range(1,unique.shape[0]):
            val = unique[i]
            ind = init_mask == val
            # print(ind.shape)
            mask[:, :, i-1][ind] = 1

        # print(np.sum(mask))

        # print(mask)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        
        if info["source"] == "lanes":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# dataset_train = lanesDataset()
# dataset_train.load_lanes("val")
# dataset_train.prepare()



def train(model):
    """Train the model."""
    # Training dataset.
    print("Preparing train dataset")

    dataset_train = lanesDataset()
    dataset_train.load_lanes("train")
    dataset_train.prepare()

    # Validation dataset
    print("Preparing val dataset")
    dataset_val = lanesDataset()
    dataset_val.load_lanes("val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1500,
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    # mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    
    lanes = np.zeros_like(image)

    color = [[255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    if mask.shape[1] == image.shape[1] and mask.shape[0] == image.shape[0]:

        for i in range(mask.shape[2]):

            # splash = np.where(mask, image, gray).astype(np.uint8)
            # print(mask.shape)

            lanes[mask[:, :, i] > 0] = color[i%len(color)]
        

        splash = cv2.addWeighted(image, 1, lanes, 1, 0)

        # print(mask)

        return splash
    else:
        return image

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        
        image = cv2.imread(image_path)
        start = time.time()

        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        print(time.time() - start)
        cv2.imshow("res", splash)
        cv2.waitKey(0)

        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # skimage.io.imsave(file_name, splash)

    elif video_path:
        # import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    # print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect laness.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
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
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments


    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = lanesConfig()
    else:
        class InferenceConfig(lanesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
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
    if args.command == "train":
        train(model)
    elif args.command == "splash":

        fimgs = glob.glob('/media/ml3/Volume/sg/sentosa_lane_label/image/*.png')
        # fimgs = glob.glob('/media/ml3/Volume/sg/sentosa/*.jpg')
        for i in range(int(len(fimgs)/2)):
            fimg = fimgs[i*2]
            print(fimg)
            detect_and_color_splash(model, image_path=fimg)


    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
