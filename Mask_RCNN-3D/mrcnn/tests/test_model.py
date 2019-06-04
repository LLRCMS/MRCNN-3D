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