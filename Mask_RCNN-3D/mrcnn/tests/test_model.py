import pytest
import numpy as np
import h5py
import os, sys

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import model
from mrcnn import config


def make_mask_3D(size, bb_coord):
    """ make mask of size `size` and put a bb at coordinate bb_coord """
    mask = np.zeros(size)
    z1, y1, x1, z2, y2, x2 = bb_coord
    mask[z1:z2, y1:y2, x1:x2] = 1
    return mask

def test_parse_image_meta_3D():
  image_id = 0
  original_image_shape = (12, 16, 16)
  image_shape = (28, 28, 28)
  window = (3, 0, 0, 24, 28, 28)
  scale = 1.75
  active_class_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

  meta = model.compose_image_meta_3D(image_id, original_image_shape, image_shape,
                          window, scale, active_class_ids)

  parsed = model.parse_image_meta_3D(np.array([meta]))

  assert parsed["image_id"] == image_id
  assert np.all(parsed["original_image_shape"] == original_image_shape)
  assert np.all(parsed["image_shape"] == image_shape)
  assert np.all(parsed["window"] == window)
  assert parsed["scale"] == scale
  assert np.all(parsed["active_class_ids"] == active_class_ids)


def test_load_image_gt_3D():
  class Dataset_3D():
    """Toy dataset for the test"""

    def __init__(self):
        f = h5py.File('../test_img/test_3d_mnist_4.hdf5', 'r')
        self.image = np.array(f['3d_mnist_4']).reshape((16,16,16))
        self.shape = self.image.shape
        self.bbox = (0, 0, 0, 10, 10, 10)
        self.num_classes = 1
        self.image_info = [{"source":"test_img/"}]
        self.source_class_ids = {"test_img/":0}
        
    def load_image(self, image_id):
        return self.image
    
    def load_mask(self, image_id):
        mask1 = np.zeros(self.shape[:3])
        mask1[...] = 1
        mask2 = np.zeros(self.shape[:3])
        mask2[:10, :20, :20] = 1
        mask3 = np.zeros(self.shape[:3])
        masks = np.stack([mask1, mask2, mask3], axis = -1)
        return masks, np.array([0, 1, 2])

  dataset_3D = Dataset_3D()
  
  class MyConfig(config.Config):
    IMAGE_MIN_DIM = 18
    IMAGE_MIN_SCALE = 1
    IMAGE_MAX_DIM = 24
    IMAGE_RESIZE_MODE = "cube"
    MINI_MASK_SHAPE = (5, 5, 5)
    
  configuration = MyConfig()
  image, image_meta, class_ids, bbox, mask = model.load_image_gt_3D(dataset_3D, configuration, 0)

  meta = model.parse_image_meta_3D(np.array([image_meta]))

  assert image.shape == (24,24,24)

  assert np.all(meta["image_id"] == [0])
  assert np.all(meta["original_image_shape"] == [16,16,16])
  assert np.all(meta["window"] == [3,3,3,21,21,21])
  assert np.all(meta["scale"] == [1.125])
  assert np.all(meta["active_class_ids"] == [[1]])

  assert np.all(bbox == [[ 3,  3,  3, 21, 21, 21], [ 3,  3,  3, 14, 21, 21]] )

  assert mask.shape == (24,24,24,2)

def test_build_rpn_targets():
  image_shape = (28, 28, 28)
  gt_class_ids = np.array([0, 1, 2, 5, 1], dtype=np.int32)
  box1 = np.array([3, 3, 3, 4, 4, 4])
  box2 = np.array([0, 0, 0, 4, 4, 4])
  box3 = np.array([0, 0, 1, 4, 4, 4])
  box4 = np.array([0, 1, 1, 4, 4, 4])
  box5 = np.array([1, 1, 1, 4, 4, 4])
  gt_boxes = np.array([box1, box2, box3, box4, box5], dtype=np.int32)
  
  scales = [12]
  ratios_xy = [0.5, 1, 2]
  ratios_xz = [0.5, 1, 2]
  shape = (6, 6, 6)
  feature_stride = 2
  anchor_stride = 1

  anchors = utils.generate_anchors_3D(scales, ratios_xy, ratios_xz, shape, feature_stride, anchor_stride)

  class my_config():
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128
    ANCHOR_MAX_IOU = 0.3
    ANCHOR_MIN_IOU = 0.7
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
      
  configu = my_config() 

  rpn_match, rpn_bbox = model.build_rpn_targets_3D(image_shape, anchors, gt_class_ids, gt_boxes, configu)

  assert rpn_match[rpn_match < 0].shape[0] + rpn_match[rpn_match > 0].shape[0] == configu.RPN_TRAIN_ANCHORS_PER_IMAGE

  assert rpn_bbox.shape == (configu.RPN_TRAIN_ANCHORS_PER_IMAGE, 6)

def test_generate_random_rois():
  image_shape = (28, 28, 28)
  count = 200
  gt_class_ids = [0, 1, 2]
  box1 = np.array([3, 3, 3, 4, 4, 4])
  box2 = np.array([0, 0, 0, 4, 4, 4])
  box3 = np.array([0, 0, 1, 4, 4, 4])
  box4 = np.array([0, 1, 1, 4, 4, 4])
  box5 = np.array([1, 1, 1, 4, 4, 4])
  boxes = np.array([box1, box2, box3, box4, box5])

  rois = model.generate_random_rois_3D(image_shape, count, gt_class_ids, boxes)

  assert rois.shape == (count, 6)

def test_build_detection_targets():
  box1 = np.array([3, 3, 3, 4, 4, 4])
  box2 = np.array([0, 0, 0, 4, 4, 4])
  box3 = np.array([0, 0, 1, 4, 4, 4])
  box4 = np.array([0, 1, 1, 4, 4, 4])
  box5 = np.array([1, 1, 1, 4, 4, 4])
  boxes = np.array([box1, box2, box3, box4, box5], dtype=np.int32)
  size = (28, 28, 28)

  mask1 = make_mask_3D(size, box1)
  mask2 = make_mask_3D(size, box2)
  mask3 = make_mask_3D(size, box3)
  mask4 = make_mask_3D(size, box4)
  mask5 = make_mask_3D(size, box5)
  masks = np.stack([mask1, mask2, mask3, mask4, mask5], axis = -1).astype(np.bool_)

  image_shape = (28, 28, 28)
  count = 200
  gt_class_ids = [0, 1, 2]

  rpn_rois = model.generate_random_rois_3D(image_shape, count, gt_class_ids, boxes)

  gt_class_ids = np.array([0, 1, 2, 5, 1], dtype=np.int32)
  gt_boxes = boxes
  gt_masks = masks

  class my_config():
      TRAIN_ROIS_PER_IMAGE = 50
      ROI_POSITIVE_RATIO = 0.33
      NUM_CLASSES = 7
      MASK_SHAPE = [28, 28, 28]
      BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
      USE_MINI_MASK = True
      MINI_MASK_SHAPE = (6, 6, 6)
      IMAGE_SHAPE = np.array([28, 28, 28])
      
  configu = my_config()

  rois, roi_gt_class_ids, bboxes, masks = model.build_detection_targets_3D(rpn_rois, gt_class_ids, gt_boxes, gt_masks, configu)

  assert rois.shape == (configu.TRAIN_ROIS_PER_IMAGE, 6)
  assert roi_gt_class_ids.shape == (50,)
  assert np.all(roi_gt_class_ids < configu.NUM_CLASSES)
  assert bboxes.shape == (configu.TRAIN_ROIS_PER_IMAGE, 
                          configu.NUM_CLASSES, 6)
  assert masks.shape == (configu.TRAIN_ROIS_PER_IMAGE, 28, 28, 28, 
                         configu.NUM_CLASSES)




def test_data_generator():

  class MyConfig():
    IMAGE_SHAPE = (24,24,24)
    RPN_ANCHOR_SCALES = [2]
    RPN_ANCHOR_RATIOS_XY = [0.5, 1, 2]
    RPN_ANCHOR_RATIOS_XZ = [0.5, 1, 2]
    BACKBONE_STRIDES = [2]
    RPN_ANCHOR_STRIDE = 1
    USE_MINI_MASK = False # TOCHANGE for another test
    RPN_TRAIN_ANCHORS_PER_IMAGE = 50
    MAX_GT_INSTANCES = 100
    NUM_CLASSES = 11
    TRAIN_ROIS_PER_IMAGE = 100
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) 
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    MINI_MASK_SHAPE = (4,4,4)
    ROI_POSITIVE_RATIO = 0.33
    ANCHOR_MAX_IOU = 0.3
    ANCHOR_MIN_IOU = 0.7
    IMAGE_MIN_DIM = 18
    IMAGE_MIN_SCALE = 1
    IMAGE_MAX_DIM = 24
    IMAGE_RESIZE_MODE = "cube"
    BACKBONE = "resnet50"
    MEAN_PIXEL = 0
    
  configu = MyConfig() 

  class Dataset_3D():
    """Toy dataset for the test"""
    def __init__(self):
        f = h5py.File('../test_img/test_3d_mnist_4.hdf5', 'r')
        self.image = np.array(f['3d_mnist_4']).reshape((16,16,16))
        self.shape = self.image.shape
        self.bbox = (0, 0, 0, 10, 10, 10)
        self.num_classes = 1
        self.image_info = [{"source":"test_img/"}]
        self.source_class_ids = {"test_img/":0}
        self.image_ids = np.array([0])
        
    def load_image(self, image_id):
        return self.image
    
    def load_mask(self, image_id):
        mask1 = np.zeros(self.shape[:3])
        mask1[...] = 1
        mask2 = np.zeros(self.shape[:3])
        mask2[:10, :20, :20] = 1
        mask3 = np.zeros(self.shape[:3])
        masks = np.stack([mask1, mask2, mask3], axis = -1)
        return masks, np.array([0, 1, 2])

  dataset_3D = Dataset_3D()

  gen = model.data_generator_3D(dataset_3D, configu)
  inputs, outputs = next(gen)
  images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = \
    inputs
  
  parsed_meta = model.parse_image_meta_3D(image_meta)

  assert images.shape == (1, 24, 24, 24)
  assert rpn_match[rpn_match < 0].shape[0] + rpn_match[rpn_match > 0].shape[0] \
    == configu.RPN_TRAIN_ANCHORS_PER_IMAGE
  assert rpn_bbox.shape == (1, configu.RPN_TRAIN_ANCHORS_PER_IMAGE, 6)
  assert gt_class_ids.shape == (1, configu.TRAIN_ROIS_PER_IMAGE)
  assert gt_masks.shape == (1, 24, 24, 24, configu.TRAIN_ROIS_PER_IMAGE)


if __name__ == "__main__":
  pass