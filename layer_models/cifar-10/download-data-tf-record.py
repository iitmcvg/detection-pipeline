#import _pickle as pickle
import pickle
import os
import re
import shutil
import tarfile
import tensorflow as tf

from datasets import dataset_utils

tf.logging.info("tensorflow version {}".format(tf.__version__))

tf.flags.DEFINE_string('CIFAR_FILENAME', 'cifar-10-python.tar.gz',
                       'Zip file to untar')
tf.flags.DEFINE_string('CIFAR_DOWNLOAD_URL', 'http://www.cs.toronto.edu/~kriz/',
                       'CIFAR 10 Download URL.')
tf.flags.DEFINE_string('CIFAR_LOCAL_FOLDER', 'cifar-10-batches-py',
                       'Local Batches folder')
tf.flags.DEFINE_string('DATA_DIR', 'cifar-10',
                       'Download folder')

FLAGS = tf.flags.FLAGS

CIFAR_FILENAME = FLAGS.CIFAR_FILENAME
CIFAR_DOWNLOAD_URL = FLAGS.CIFAR_DOWNLOAD_URL + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = FLAGS.CIFAR_LOCAL_FOLDER

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _download_and_extract(data_dir):
    tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir, CIFAR_DOWNLOAD_URL)
    tarfile.open(os.path.join(data_dir, CIFAR_FILENAME), 'r:gz').extractall(data_dir)


def  _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names

def _read_pickle_from_file(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data_dict = u.load()
    return data_dict

def _convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = _read_pickle_from_file(input_file)
            data = data_dict['data']
            labels =  data_dict['labels']
            num_entries_in_batch = len(labels)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
            feature={
            'image': _bytes_feature(data[i].tobytes()),
            'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def main(_):
    data_dir=FLAGS.DATA_DIR
    _download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)

    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode+'.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
    # Convert to tf.train.Example and write to TFRecords.
        _convert_to_tfrecord(input_files, output_file)

if __name__=='__main__':
    tf.app.run(main=main)
