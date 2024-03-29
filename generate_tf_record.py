"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
#
#!set PYTHONPATH=...\python\models;...\python\models\slim
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image

from object_detection.utils import dataset_util
from collections import namedtuple

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'angle':
        return 1
    if row_label == 'dot':
        return 2
    if row_label == 'line':
        return 3
    if row_label == 'square':
        return 4
    if row_label == 'z':
        return 5
    if row_label == 'color_wok':
        return 6
    if row_label == 'craft_wok':
        return 7
    if row_label == 'craft_wokw':
        return 7
    if row_label == 'craf_wok':
        return 7
    if row_label == 'cup':
        return 8
    if row_label == 'salad':
        return 9
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
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(os.path.basename(filename)),
        'image/source_id': dataset_util.bytes_feature(os.path.basename(filename)),
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


from sklearn.model_selection import train_test_split

def write_data_to_tf(examples, filename, path):
    #(examples, filename, path) = (x_train, 'goods_train.record', path)
    writer = tf.python_io.TFRecordWriter(filename)
    grouped = split(examples, 'filename')
    for group in grouped:
        #print(group)
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    path = os.getcwd()#os.path.join(os.getcwd(), 'autorec_numbers')
    data = pd.read_csv('all_goods.csv')
    
    x_train, x_test, _, _ = train_test_split(data, data['class'], test_size=0.33)  
    print(x_test['class'].unique() ) 
    
    write_data_to_tf(x_train, 'goods_train.record', path)
    write_data_to_tf(x_test, 'goods_test.record', path)

if __name__ == '__main__':
    tf.app.run()