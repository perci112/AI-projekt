from object_detection.utils import label_map_util
import tensorflow as tf

# Monkey patch to fix tf.gfile issue
label_map_util.tf.gfile = tf.io.gfile

label_map = label_map_util.load_labelmap('dupa.pbtxt')
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=90, use_display_name=True)
with open('dupa.pbtxt', 'r') as f:
    print(f.read())
