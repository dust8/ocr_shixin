import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
from PIL import Image


def load_image(path) -> np.ndarray:
    """load a picture as numpy.ndarray"""
    return np.array(Image.open(path))


def save_image(src, path):
    """save image"""
    img = Image.fromarray(src)
    img.save(path)


def rgb2gray(src) -> np.ndarray:
    """把 rgb 图片转换为灰度图"""
    return np.dot(src[..., :3], [0.299, 0.587, 0.114])


def threshold(src, thresh=127, maxval=255, thresh_type=0) -> np.ndarray:
    """二值化图片

    0 THRESH_BINARY
    1 THRESH_BINARY_INV

    https://blog.csdn.net/a19990412/article/details/81172426
    """
    if thresh_type == 0:
        src[src > thresh] = maxval
        src[src <= thresh] = 0
    elif thresh_type == 1:
        src[src > thresh] = 0
        src[src <= thresh] = maxval
    else:
        raise ValueError("thresh_type is invaild!")

    return src


def mean_threshold(src):
    """取平均值确定二值化的阈值"""
    return np.mean(src)


def adaptive_threshold(src, maxval=255, adaptive_method="mean"):
    """自适应阈值"""
    if adaptive_method == "mean":
        thresh = mean_threshold(src)
        dst = threshold(src, thresh)
    else:
        raise ValueError("adaptive method is invalid!")
    return dst


def pad_image(array) -> np.ndarray:
    """pads image height equal width"""
    height, width = array.shape

    pad_size = abs(height - width) // 2
    remainder = abs(height - width) % 2

    if height < width:
        pad_width = ((pad_size, pad_size + remainder), (0, 0))
    else:
        pad_width = ((0, 0), (pad_size, pad_size + remainder))

    return np.pad(array, pad_width, "constant", constant_values=255)


def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-15, 15), reshape=False, cval=1.0)
    # 防止超过 1.0, 导致变灰
    image[image > 1] = 1.0
    return image


def zoom_image(array, zoom=1.0) -> np.ndarray:
    array = ndimage.zoom(array, zoom)
    # 防止超过 255.0, 导致变灰
    array[array > 255] = 255
    return array


def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label


def random_brightness_image(image, label):
    return tf.image.random_brightness(image, max_delta=0.3), label


def random_contrast_image(image, label):
    return tf.image.random_contrast(image, 0.8, 1.2), label


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image_string, label) -> str:
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        "height": _int64_feature([image_shape[0]]),
        "width": _int64_feature([image_shape[1]]),
        "image_raw": _bytes_feature([image_string]),
        "label": _bytes_feature([label.encode()]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def load_class_table(filename) -> tf.lookup.StaticHashTable:
    class_table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            filename, tf.string, 0, tf.int64, -1, delimiter="\n"
        ),
        -1,
    )

    return class_table


def load_raw_dataset(file_pattern) -> tf.train.Example:
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        yield example
