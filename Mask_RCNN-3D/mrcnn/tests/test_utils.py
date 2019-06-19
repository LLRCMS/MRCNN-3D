import pytest
import numpy as np
import h5py
import os, sys

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import model
from mrcnn import config



def _volume(box):
  """ compute volume of a 3D box"""
  return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

def test_compute_iou_3D():
  # make some test 3D boxes
  box =  np.array([0, 0, 0, 2, 2, 2]) # base box, vol: 8
  box_area = _volume(box)

  box1 = np.array([3, 3, 3, 4, 4, 4]) # no intersection
  box2 = np.array([0, 0, 0, 4, 4, 4]) # vol: 64 inter: 8, union: 64
  box3 = np.array([0, 0, 1, 4, 4, 4]) # vol: 48 inter: 4, union: 52 
  box4 = np.array([0, 1, 1, 4, 4, 4]) # vol: 36 inter: 2, union: 42
  box5 = np.array([1, 1, 1, 4, 4, 4]) # vol: 27 inter: 1, union: 34

  boxes_area = np.array([_volume(box1), _volume(box2), _volume(box3), _volume(box4), _volume(box5)])
  boxes = np.array([box1, box2, box3, box4, box5])

  iou = utils.compute_iou_3D(box, boxes, box_area, boxes_area)
  assert iou[0] == 0
  assert iou[1] == 8/64
  assert iou[2] == 4/52
  assert iou[3] == 2/42
  assert iou[4] == 1/34

def test_compute_overlaps_3D():
  box1 =  np.array([0, 0, 0, 2, 2, 2]) # base box,  vol: 8
  box2 =  np.array([1, 1, 1, 2, 2, 2]) # base box2, vol: 4

  box1_b = np.array([3, 3, 3, 4, 4, 4]) # no intersection
  box2_b = np.array([0, 0, 0, 4, 4, 4]) # vol: 64 inter: 8, union: 64
  box3_b = np.array([0, 0, 1, 4, 4, 4]) # vol: 48 inter: 4, union: 52 
  box4_b = np.array([0, 1, 1, 4, 4, 4]) # vol: 36 inter: 2, union: 42
  box5_b = np.array([1, 1, 1, 4, 4, 4]) # vol: 27 inter: 1, union: 34

  boxes1 = np.array([box1, box2])
  boxes2 = np.array([box1_b, box2_b, box3_b, box4_b, box5_b])

  box1_vol = _volume(box1)
  box2_vol = _volume(box2)

  boxes_vol = np.array([_volume(box1_b), _volume(box2_b), _volume(box3_b), _volume(box4_b), _volume(box5_b)])

  overlaps = utils.compute_overlaps_3D(boxes1, boxes2)

  iou_box1 = utils.compute_iou_3D(box1, boxes2, box1_vol, boxes_vol)
  iou_box2 = utils.compute_iou_3D(box2, boxes2, box2_vol, boxes_vol)

  assert np.all(overlaps[0, :] == iou_box1)
  assert np.all(overlaps[1, :] == iou_box2)

def make_mask_3D(size, bb_coord):
    """ make mask of size `size` and put a bb at coordinate bb_coord """
    mask = np.zeros(size)
    z1, y1, x1, z2, y2, x2 = bb_coord
    mask[z1:z2, y1:y2, x1:x2] = 1
    return mask

def test_extract_bboxes_3D():
  size = (4, 4, 4)

  coord1 = (0, 0, 0, 2, 2, 2)
  coord2 = (0, 2, 0, 1, 4, 2)
  coord3 = (1, 2, 0, 2, 4, 2)
  
  mask1 = make_mask_3D(size, coord1)
  mask2 = make_mask_3D(size, coord2)
  mask3 = make_mask_3D(size, coord3)

  # stack the masks along the last axe
  masks = np.stack([mask1, mask2, mask3], axis = -1)

  boxes = utils.extract_bboxes_3D(masks)

  assert np.all([list(x) for x in (coord1, coord2, coord3)] == boxes)


def test_resize_image_3D():
  # open test 3D MNIST file
  f = h5py.File('../test_img/test_3d_mnist_4.hdf5', 'r')
  x4 = np.array(f['3d_mnist_4'])
  # reshape it to cube (it is flattened in the file)
  x4 = x4.reshape((16,16,16))
  x4_cropped = x4[:12, ...]

  x4_resized, window, scale, padding, crop  = utils.resize_image_3D(x4, min_dim=28, max_dim=28)
  x4_cropped_resized, window_cropped, scale_cropped, padding_cropped, crop_cropped  = utils.resize_image_3D(x4_cropped, min_dim=28, max_dim=28)

  assert x4_resized.shape == (28, 28, 28)
  assert x4_cropped_resized.shape == (28, 28, 28)

  assert window == (0, 0, 0, 28, 28, 28)
  assert window_cropped == (3, 0, 0, 24, 28, 28)

  assert np.all(padding == [(0,0), (0,0), (0,0)])
  assert np.all(padding_cropped == [(3,4), (0,0), (0,0)])

  assert scale == 1.75
  assert scale_cropped == 1.75

def test_resize_mask_3D():
  # TODO: take care of size of mask (fourth dim) and padding
  size = (4, 4, 4)
  coord1 = (0, 0, 0, 2, 2, 2)
  coord2 = (0, 2, 0, 1, 4, 2)
  coord3 = (1, 2, 0, 2, 4, 2)

  mask1 = make_mask_3D(size, coord1)
  mask2 = make_mask_3D(size, coord2)
  mask3 = make_mask_3D(size, coord3)

  masks = np.stack([mask1, mask2, mask3], axis = -1)
  padding =  [(1,2), (3,4), (5,6)]

  masks_resized = utils.resize_mask_3D(masks, 1.25, padding)

  assert masks_resized.shape == (8, 12, 16, 3)

  # resize_mask_3D only works with stack of masks, tensor of rank 4
  with pytest.raises(RuntimeError) as e_info:
    masks_resized = utils.resize_mask_3D(mask1, 1.25, padding)

  error_message = "sequence argument must have length equal to input rank"
  assert error_message in str(e_info.value)

def test_box_refinement_3D():

  # different centers, same dimensions
  box1    = [3, 4, 5, 6, 7, 8]
  gt_box1 = [5, 6, 7, 8, 9, 10]

  # identical
  box2    = [4, 4, 4, 13, 12, 15]
  gt_box2 = [4, 4, 4, 13, 12, 15]

  # different everything
  box3    = [3, 4, 5, 6, 7, 8]
  gt_box3 = [5, 6, 7, 10, 9, 8]

  boxes = np.stack([box1, box2, box3])
  gt_boxes = np.stack([gt_box1, gt_box2, gt_box3])

  result = utils.box_refinement_3D(boxes, gt_boxes)

  # numerical results for the following asserts
  two_third = 2/3
  center = [(7.5 - 4.5) / 3, (7.5 - 5.5) / 3, (7.5 - 6.5) / 3]
  log_dim = np.log([5/3, 3/3, 1/3])

  assert np.all(result[:, 0] == pytest.approx([two_third, 0, center[0]]))
  assert np.all(result[:, 1] == pytest.approx([two_third, 0, center[1]]))
  assert np.all(result[:, 2] == pytest.approx([two_third, 0, center[2]]))
  assert np.all(result[:, 3] == pytest.approx([0, 0, log_dim[0]]))
  assert np.all(result[:, 4] == pytest.approx([0, 0, log_dim[1]]))
  assert np.all(result[:, 5] == pytest.approx([0, 0, log_dim[2]]))

def test_generate_anchors_3D():

  scales = [32, 64, 128]
  ratios_xy = [0.5, 1, 2]
  ratios_xz = [0.5, 1, 2]
  shape = (25, 25, 25)
  feature_stride = 32
  anchor_stride = 1

  boxes = utils.generate_anchors_3D(scales, ratios_xy, ratios_xz, shape, feature_stride, anchor_stride)

  assert boxes.shape == (25*25*25*27, 6)


def test_compute_overlaps_masks_3D():
  coord1 =  np.array([0, 0, 0, 2, 2, 2]) # base box,  vol: 8
  coord2 =  np.array([1, 1, 1, 2, 2, 2]) # base box2, vol: 4

  coord1_b = np.array([3, 3, 3, 4, 4, 4]) # no intersection
  coord2_b = np.array([0, 0, 0, 4, 4, 4]) # vol: 64 inter: 8, union: 64
  coord3_b = np.array([0, 0, 1, 4, 4, 4]) # vol: 48 inter: 4, union: 52 
  coord4_b = np.array([0, 1, 1, 4, 4, 4]) # vol: 36 inter: 2, union: 42
  coord5_b = np.array([1, 1, 1, 4, 4, 4]) # vol: 27 inter: 1, union: 34

  size = (4,4,4)

  mask1 = make_mask_3D(size, coord1)
  mask2 = make_mask_3D(size, coord2)

  mask1_b = make_mask_3D(size, coord1_b)
  mask2_b = make_mask_3D(size, coord2_b)
  mask3_b = make_mask_3D(size, coord3_b)
  mask4_b = make_mask_3D(size, coord4_b)
  mask5_b = make_mask_3D(size, coord5_b)

  masks1 = np.stack([mask1, mask2], axis=-1)
  masks2 = np.stack([mask1_b, mask2_b, mask3_b, mask4_b, mask5_b], axis=-1)

  overlaps = utils.compute_overlaps_masks_3D(masks1, masks2)

  # same results as the corresponding bboxes test, above
  overlaps1 = [0, 0.125, 0.07692308, 0.04761905, 0.02941176]
  overlaps2 = [0, 0.015625, 0.02083333, 0.02777778, 0.03703704]

  assert overlaps[0, :] == pytest.approx(overlaps1)
  assert overlaps[1, :] == pytest.approx(overlaps2)


def test_non_max_suppression_3D():

  box1 = np.array([3, 3, 3, 4, 4, 4]) # no intersection
  box2 = np.array([0, 0, 0, 4, 4, 4]) # vol: 64 inter: 8, union: 64
  box3 = np.array([0, 0, 1, 4, 4, 4]) # vol: 48 inter: 4, union: 52 
  box4 = np.array([0, 1, 1, 4, 4, 4]) # vol: 36 inter: 2, union: 42
  box5 = np.array([1, 1, 1, 4, 4, 4]) # vol: 27 inter: 1, union: 34

  boxes = np.array([box1, box2, box3, box4, box5])

  scores = np.array([0.2, 0.5, 0.6, 0.7, 0.8])

  threshold = 0.3


  nms = utils.non_max_suppression_3D(boxes, scores, threshold)

  assert np.all(nms == [4, 0])

if __name__ == "__main__":
  test_non_max_suppression_3D()