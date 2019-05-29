# GG ???
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3= 1

import os
import sys
import random
import math
import re
import time
import numpy as np
import shapes as shps

# Root directory of the project\n",
ROOT_DIR = os.path.abspath("../../")
MODEL_PATH = "/data_CMS/cms/grasseau/HAhRD/Mask-RCNN/samples"
import mrcnn.model as modellib
from mrcnn import visualize

# GG : Limits the numbet of threads used & use 
# legacy threads
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
## COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = os.path.join(MODEL_PATH, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = shps.ShapesConfig()
config.display()

# Training dataset
dataset_train = shps.ShapesDataset()
dataset_train.load_shapes(10, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()
    
# Validation dataset
dataset_val = shps.ShapesDataset()
dataset_val.load_shapes(1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    print( "image ", image.dtype , image.shape )
    print( "mask ",  mask.dtype, mask.shape, mask )
    print( "class ", class_ids, class_ids.dtype )

## Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

## Training
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train( dataset_train, dataset_val, 
             learning_rate=config.LEARNING_RATE, 
             epochs=1, 
             layers='heads')

# pass a regular expression to select which layers to
# train by name pattern.
model.train( dataset_train, dataset_val, 
             learning_rate=config.LEARNING_RATE / 10,
             epochs=2, 
             layers="all")


