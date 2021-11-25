from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == 'd6-1':
        return 1
    elif row_label == 'd6-2':
        return 2
    elif row_label == 'd6-3':
        return 3
    elif row_label == 'd6-4':
        return 4
    elif row_label == 'd6-5':
        return 5
    elif row_label == 'd6-6':
        return 6
    elif row_label == 'd6p-1':
        return 7
    elif row_label == 'd6p-2':
        return 8
    elif row_label == 'd6p-3':
        return 9
    elif row_label == 'd6p-4':
        return 10
    elif row_label == 'd6p-5':
        return 11
    elif row_label == 'd6p-6':
        return 12
    elif row_label == 'd8p-1':
        return 13
    elif row_label == 'd8p-2':
        return 14
    elif row_label == 'd8p-3':
        return 15
    elif row_label == 'd8p-4':
        return 16
    elif row_label == 'd8p-5':
        return 17
    elif row_label == 'd8p-6':
        return 18
    elif row_label == 'd8p-7':
        return 19
    elif row_label == 'd8p-8':
        return 20
    elif row_label == 'd10-1':
        return 21
    elif row_label == 'd10-2':
        return 22
    elif row_label == 'd10-3':
        return 23
    elif row_label == 'd10-4':
        return 24
    elif row_label == 'd10-5':
        return 25
    elif row_label == 'd10-6':
        return 26
    elif row_label == 'd10-7':
        return 27
    elif row_label == 'd10-8':
        return 28
    elif row_label == 'd10-9':
        return 29
    elif row_label == 'd10-10':
        return 30
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()