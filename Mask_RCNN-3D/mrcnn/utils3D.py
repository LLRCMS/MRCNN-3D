"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [depth, height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[..., i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=(0,1)))[0]
        vertical_indicies = np.where(np.any(m, axis=(0,2)))[0]
        depthical_indicies = np.where(np.any(m, axis=(1,2)))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = depthical_indicies[[0, -1]]
            # x2, y2 and z2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            z2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([z1, y1, x1, z2, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_vol, boxes_vol):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_vol: float. the volume of 'box'
    boxes_vol: array of length boxes_count.

    Note: the volumes are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_vol + boxes_vol[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_iou_graph(box, boxes, box_vol, boxes_vol):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_vol: float. the volume of 'box'
    boxes_vol: array of length boxes_count.

    Note: the volumes are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = tf.maximum(box[0], boxes[:, 0])
    z2 = tf.minimum(box[3], boxes[:, 3])
    y1 = tf.maximum(box[1], boxes[:, 1])
    y2 = tf.minimum(box[4], boxes[:, 4])
    x1 = tf.maximum(box[2], boxes[:, 2])
    x2 = tf.minimum(box[5], boxes[:, 5])
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(z2 - z1, 0)
    union = box_vol + boxes_vol[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Volumes of anchors and GT boxes
    vol1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    vol2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou_3D(box2, boxes1, vol2[i], vol1)
    return overlaps

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Depth, Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their volumes
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    vol1 = np.sum(masks1, axis=0)
    vol2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.transpose(), masks2)
    union = vol1[:, None] + vol2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Notice that (z2, y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    vol = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou_3D(boxes[i], boxes[ixs[1:]], vol[i], vol[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def non_max_suppression_graph(boxes, scores, max_output_size, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Notice that (z2, y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0

    # Compute box areas
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    vol = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = tf.argsort(scores)[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou_graph_3D(boxes[i], boxes[ixs[1:]], vol[i], vol[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = tf.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = tf.gather(ixs, remove_ixs)
        ixs = tf.gather(ixs, 0)
    return tf.constant(pick[:max_output_size])

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Note that (z2, y2, x2) is outside the box.
    deltas: [N, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= np.exp(deltas[:, 3])
    height *= np.exp(deltas[:, 4])
    width *= np.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([z1, y1, x1, z2, y2, x2], axis=1)

def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = tf.log(gt_depth / depth)
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dz, dy, dx, dd, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]. (z2, y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = np.log(gt_depth / depth)
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dz, dy, dx, dd, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

# class Dataset(object):

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="cube"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        cube: Resize and pad with zeros to get a cube image
            of size [max_dim, max_dim, max_dim].
        pad64: Pads width, height and depth with zeros to make them multiples 
               of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (z1, y1, x1, z2, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2, z2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image 
    [(front, back), (top, bottom), (left, right), (0,0)]
    TODO: check if to take into account (or not) a fourth dimension for the
    channels
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (z1, y1, x1, z2, y2, x2) and default scale == 1.
    d, h, w = image.shape[:3]
    window = (0, 0, 0, d, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(d, h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "cube":
        image_max = max(d, h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, 
                       (round(d * scale), round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "cube":
        # Get new height and width
        d, h, w = image.shape[:3]
        front_pad = (max_dim - d) // 2
        back_pad = max_dim - d - front_pad
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(front_pad, back_pad), (top_pad, bottom_pad), 
                   (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (front_pad, top_pad, left_pad, 
                  d + front_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        d, h, w = image.shape[:3]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Depth
        if d % 64 > 0:
            max_d = d - (d % 64) + 64
            front_pad = (max_d - d) // 2
            back_pad = max_d - d - front_pad
        else:
            front_pad = back_pad = 0
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(front_pad, back_pad), (top_pad, bottom_pad), 
                   (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (front_pad, top_pad, left_pad, 
                  d + front_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        d, h, w = image.shape[:3]
        z = random.randint(0, (d - min_dim))
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (z, y, x, min_dim, min_dim, min_dim)
        image = image[z:z + min_dim, y:y + min_dim, x:x + min_dim]
        window = (0, 0, 0, min_dim, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(front, back), (top, bottom), (left, right), (0,0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, scale, 1], order=0)
    if crop is not None:
        z, y, x, d, h, w = crop
        mask = mask[z:z + d, y:y + h, x:x + w]
    else:
        padding.append((0, 0))
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks_3D()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, :, i].astype(bool)
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        m = m[z1:z2, y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, :, i] = np.around(m).astype(np.bool)
    return mini_mask

def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask_3D().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:3] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, :, i]
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        d = z2 - z1
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (d, h, w))
        mask[z1:z2, y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask

############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios_xy, ratios_xz, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios_xy: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    ratios_xz: 1D array of anchor ratios of width/depth. Example: [0.5, 1, 2]
    shape: [depth, eight, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios_xy, ratios_xz = np.meshgrid(np.array(scales), np.array(ratios_xy), np.array(ratios_xz))
    scales = scales.flatten()
    ratios_xy = ratios_xy.flatten()
    ratios_xz = ratios_xz.flatten()

    # Enumerate depths, heights and widths from scales and the 2 ratios
    depths = scales * np.sqrt(ratios_xz)
    heights = scales / np.sqrt(ratios_xy)
    widths = scales * np.sqrt(ratios_xy)

    # Enumerate shifts in feature space
    shifts_z = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_y = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[2], anchor_stride) * feature_stride
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, heights and depths
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (z, y, x) and a list of (d, h, w)
    box_centers = np.stack(
        [box_centers_z, box_centers_y, box_centers_x], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_depths, box_heights, box_widths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (z1, y1, x1, z2, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes

def generate_pyramid_anchors(scales, ratios_xy, ratios_xz, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (z1, y1, x1, z2, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (z1, y1, x1, z2, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors_3D(scales[i], ratios_xy, ratios_xz, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

