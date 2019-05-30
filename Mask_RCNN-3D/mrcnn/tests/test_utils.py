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