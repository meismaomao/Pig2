from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import dataset_utils

import tensorflow as tf


slim = tf.contrib.slim

_FILE_PATTERN = 'pigs_%s_*.tfrecord'
_NUM_CLASSES = 30

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 29',
}

