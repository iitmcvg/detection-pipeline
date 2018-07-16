import tensorflow as tf
from nets import cifarnet
import shutil

# Module imports
import core.preprocessing_factory as preprocess
import core.data_augment as data_augment

# Parsing Flags
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')


tf.flags.DEFINE_string('model_name', 'cifar10-cnn-model',
                       'Model Name.')

tf.flags.DEFINE_integer('batch_size', 200,
                       'Batch size to be used.')
tf.flags.DEFINE_integer('max_steps', 1000,
                       'Maximum train steps.')
tf.flags.DEFINE_integer('eval_steps', 200,
                       'Maximum train steps.')
tf.flags.DEFINE_integer('save_checkpoints_steps', 100,
                       'Step periodicity to save checkpoints.')
tf.flags.DEFINE_integer('tf_random_seed', 19851211,
                       'Random Seed.')
tf.flags.DEFINE_string('model_name', 'cifar10-cnn-model',
                       'Model Name.')
tf.flags.DEFINE_boolean('use_checkpoint', False,
                       'Restore from a checkpoint.')

FLAGS = flags.FLAGS

pipeline_config=config_utils.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)

# Image params
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10


def parse_record(serialized_example):
    features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
    image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

def  generate_input_fn (file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames=file_names)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Transformation
        dataset = dataset.map(parse_record)
        dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))

        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2 * batch_size)

        images, labels = dataset.make_one_shot_iterator().get_next()

        features = {'images': images}
        return features, labels

    return _input_fn

def get_feature_columns():
    feature_columns = {
    'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
    }
    return feature_columns

def model_fn(features, labels, mode, params):
    # Create the input layers from the features
    feature_columns = list(get_feature_columns().values())

    images = tf.feature_column.input_layer(
    features=features, feature_columns=feature_columns)

    images = tf.reshape(
    images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

    # Calculate logits through CNN
    with tf.contrib.slim.arg_scope(cifarnet.cifarnet_arg_scope()):
        logits, end_points = cifarnet.cifarnet(images,is_training=(mode==tf.estimator.ModeKeys.TRAIN))

    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        with tf.name_scope("loss"):
            label_indices = tf.argmax(input=labels, axis=1)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits)
            tf.summary.scalar('cross_entropy', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:

        with tf.name_scope("optimiser"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

def serving_input_fn():
    receiver_tensor = {'images': tf.placeholder(
    shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

model_dir = 'trained_models/{}'.format(FLAGS.model_name)
train_data_files = ['cifar-10/train.tfrecords']
valid_data_files = ['cifar-10/validation.tfrecords']
test_data_files = ['cifar-10/eval.tfrecords']

run_config = tf.estimator.RunConfig(
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  tf_random_seed=FLAGS.tf_random_seed,
  model_dir=model_dir
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# There is another Exporter named FinalExporter
exporter = tf.estimator.LatestExporter(
  name='Servo',
  serving_input_receiver_fn=serving_input_fn,
  assets_extra=None,
  as_text=False,
  exports_to_keep=5)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

tf.logging.set_verbosity(tf.logging.INFO)

train_spec = tf.estimator.TrainSpec(
  input_fn=generate_input_fn(file_names=train_data_files,
                             mode=tf.estimator.ModeKeys.TRAIN,
                             batch_size=FLAGS.batch_size),
  max_steps=FLAGS.max_steps)

eval_spec = tf.estimator.EvalSpec(
  input_fn=generate_input_fn(file_names=valid_data_files,
                             mode=tf.estimator.ModeKeys.EVAL,
                             batch_size=FLAGS.batch_size),
  steps=FLAGS.eval_steps, exporters=exporter)

if not FLAGS.use_checkpoint:
  print("Removing previous artifacts...")
  shutil.rmtree(model_dir, ignore_errors=True)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
