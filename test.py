# /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

A = tf.ones([5, 4], dtype=tf.float32)
B = tf.constant([1, 2, 1, 3, 3], dtype=tf.int32)
w = tf.ones([5], dtype=tf.float32)

D = tf.contrib.legacy_seq2seq.sequence_loss_by_example([A], [B], [w])

with tf.Session() as sess:
    print(sess.run(D))