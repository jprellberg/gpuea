import datetime
import logging
import os
import random
import string
import sys

import numpy as np
import pathlib
import tensorflow as tf


def unique_string():
    return '{}.{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ'),
                          ''.join(random.choice(string.ascii_uppercase) for _ in range(4)))


def create_unique_dir(root=None):
    filename = unique_string()
    if root is not None:
        filename = os.path.join(root, filename)
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    return filename


def get_logger(output_file='log.txt'):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

        # Create file handler
        if output_file is not None:
            fh = logging.FileHandler(output_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Create console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def create_reset_metric(metric, **metric_args):
    with tf.variable_scope(None, 'reset_metric') as scope:
        metric_op, update_op = metric(**metric_args)
    vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope._name_scope)
    reset_op = tf.variables_initializer(vars, name='reset')
    return metric_op, update_op, reset_op


def acc(logits, labels):
    with tf.name_scope('accuracy'):
        pred = tf.nn.softmax(logits)
        pred = tf.argmax(pred, axis=1)
        pred = tf.one_hot(pred, depth=tf.shape(labels)[1])
        return create_reset_metric(tf.metrics.accuracy, labels=labels, predictions=pred, weights=labels)


def confusion_matrix(logits, labels, num_classes):
    with tf.name_scope('confusion_matrix'):
        pred = tf.nn.softmax(logits)
        pred = tf.argmax(pred, axis=1)
        label = tf.argmax(labels, axis=1)
        return tf.confusion_matrix(label, pred, num_classes)


def normalize_feature(x, mean, std):
    with tf.name_scope('normalize_feature'):
        x = tf.cast(x, tf.float32)
        x = (x - mean) / std
        return x


def normalize_label(y, classes):
    with tf.name_scope('normalize_label'):
        y = tf.cast(y, tf.int32)
        y = tf.one_hot(y, depth=classes)
        y = tf.reshape(y, [classes])
        return y


def get_trainable_var_count():
    return int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))


class AugmentImages:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x, y):
        shape = tf.shape(x)
        x = tf.image.resize_image_with_crop_or_pad(x, self.pad, self.pad)
        x = tf.random_crop(x, shape)
        x = tf.image.random_flip_left_right(x)
        return x, y


def augment_images(pad):
    return AugmentImages(pad)


class NumpyImageDataset:
    def __init__(self, data, num_classes, buffer_size=10000, prefetch=1):
        self.num_classes = num_classes
        # Load numpy data
        (x, y), (x_test, y_test) = data
        split = int(len(x) * 0.9)
        order = np.random.RandomState(1).permutation(len(x))
        x_train = x[order][:split]
        y_train = y[order][:split]
        x_val = x[order][split:]
        y_val = y[order][split:]
        # Assign as tuples of (features, label)
        self.train = x_train, y_train
        self.val = x_val, y_val
        self.test = x_test, y_test
        # Calculate mean and std for normalization
        self.mean = np.mean(self.train[0], axis=0)
        self.std = 255.

        self.buffer_size = buffer_size
        self.prefetch = prefetch

    def input(self, source, batch_size, *maps):
        if source == 'train':
            source = self.train
        elif source == 'val':
            source = self.val
        elif source == 'test':
            source = self.test
        else:
            raise AssertionError

        dataset = tf.data.Dataset.from_tensor_slices(source)
        dataset = dataset.map(lambda x, y: (normalize_feature(x, self.mean, self.std), normalize_label(y, self.num_classes)))
        for fn in maps:
            dataset = dataset.map(fn)
        return (dataset
                .shuffle(self.buffer_size)
                .repeat()
                .batch(batch_size)
                .prefetch(self.prefetch))


class Cifar10(NumpyImageDataset):
    def __init__(self, buffer_size=10000, prefetch=1):
        super().__init__(tf.keras.datasets.cifar10.load_data(), 10, buffer_size, prefetch)


class Cifar100(NumpyImageDataset):
    def __init__(self, buffer_size=10000, prefetch=1):
        super().__init__(tf.keras.datasets.cifar100.load_data(), 100, buffer_size, prefetch)


class Mnist(NumpyImageDataset):
    def __init__(self, buffer_size=10000, prefetch=1):
        super().__init__(self._data(), 10, buffer_size, prefetch)

    def _data(self):
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Add missing channel dimension to MNIST data
        x = x.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        return (x, y), (x_test, y_test)


class FashionMnist(NumpyImageDataset):
    def __init__(self, buffer_size=10000, prefetch=1):
        super().__init__(self._data(), 10, buffer_size, prefetch)

    def _data(self):
        # We use the keras dataset instead of tf.keras because tf 1.7 doesn't yet include it in tf.keras
        import keras
        (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        # Add missing channel dimension to MNIST data
        x = x.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        return (x, y), (x_test, y_test)
