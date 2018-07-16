import tensorflow as tf
from nets import cifarnet

# Parsing Flags
tf.flags.DEFINE_integer('batch_size', 200,
                       'Batch size to be used.')
tf.flags.DEFINE_integer('max_steps', 1000,
                       'Maximum train steps.')
tf.flags.DEFINE_integer('save_checkpoints_steps', 100,
                       'Step periodicity to save checkpoints.')
tf.flags.DEFINE_integer('tf_random_seed', 19851211,
                       'Random Seed.')
tf.flags.DEFINE_string('model_name', 'cifar10-cnn-model',
                       'Model Name.')
tf.flags.DEFINE_boolean('use_checkpoint', False,
                       'Restore from a checkpoint.')

FLAGS = tf.flags.FLAGS

# Image params
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10

class cifar10(object):
    '''
    Tf Estimator structure for cifar 10
    '''
    def __init__(self):
        IMAGE_HEIGHT = 32
        IMAGE_WIDTH = 32
        IMAGE_DEPTH = 3
        NUM_CLASSES = 10

    def parse_record(self,serialized_example):
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

    def preprocess_image(self,image, is_training=False):
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

    def  generate_input_fngenerat (self,file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
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

    def get_feature_columns(self):
        feature_columns = {
        'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
        }
        return feature_columns

    def model_fn(self,features, labels, mode, params):
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

    def serving_input_fn(self):
        receiver_tensor = {'images': tf.placeholder(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
        features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
