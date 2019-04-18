"""
Builds the UNet model model as described in paper:
"Anime Sketch Colowing with Swish-Gated Residual U-Net"
"""
import tensorflow as tf


def Conv2DLReLUBase(conv_func, inputs, filters, kernel_size=2, strides=1, padding='SAME'):
    layer = conv_func(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        stride=strides,
        normalizer_fn=tf.contrib.layers.layer_norm,
        activation_fn=None,
        padding=padding)
    layer = tf.nn.leaky_relu(layer)
    return layer


def Conv2DLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=tf.contrib.layers.conv2d, *args, **kwargs)


def Conv2DTransposeLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=tf.contrib.layers.conv2d_transpose, *args, **kwargs)


def SwishMod(inputs, name='SWISH'):
    with tf.variable_scope(name):
        filters = inputs.get_shape().as_list()[-1]
        conv = tf.contrib.layers.conv2d(inputs, num_outputs=filters, kernel_size=3,
                                    normalizer_fn=tf.contrib.layers.layer_norm,
                                    activation_fn=None, padding='SAME')
        swished = tf.multiply(inputs, tf.sigmoid(conv))
    return swished


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('VAR_'+var.name.replace(':','_')):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class SGRU(object):


    def __init__(self, summarize=False):

        self.image_bw = tf.placeholder(tf.float32, shape=[1, None, None, 1], name='img_bw')
        inputs = self.image_bw
        self.acts = []

        with tf.variable_scope('SGRU_MODEL'):
            acts = []
            inputs, conv1 = self._swish_gated_block('SGB_1', inputs, 96, conv1x1=False)
            a1 = inputs
            c1 = conv1
            inputs, conv2 = self._swish_gated_block('SGB_2', inputs, 192)
            a2 = inputs
            c2 = conv2
            inputs, conv3 = self._swish_gated_block('SGB_3', inputs, 288)
            a3 = inputs
            c3 = conv3
            inputs, conv4 = self._swish_gated_block('SGB_4', inputs, 384)
            a4 = inputs
            c4 = conv4
            inputs, conv5 = self._swish_gated_block('SGB_5', inputs, 480)
            a5 = inputs
            c5 = conv5
            a_list = [a1,a2,a3,a4,a5]
            acts.append(a_list)
            c_list = [c1,c2,c3,c4,c5]
            acts.append(c_list)

            swish1 = SwishMod(conv1, 'SWISH_1')
            s1 = swish1
            swish2 = SwishMod(conv2, 'SWISH_2')
            s2 = swish2
            swish3 = SwishMod(conv3, 'SWISH_3')
            s3 = swish3
            swish4 = SwishMod(conv4, 'SWISH_4')
            s4 = swish4
            swish5 = SwishMod(conv5, 'SWISH_5')
            s5 = swish5
            s_list = [s1,s2,s3,s4,s5]
            acts.append(s_list)

            inputs, sgbc = self._swish_gated_block('SGB_5_up', inputs, 512, cat=swish5)
            sgb5 = inputs
            sgbc5 = sgbc
            inputs, sgbc = self._swish_gated_block('SGB_4_up', inputs, 480, cat=swish4)
            sgb4 = inputs
            sgbc4 = sgbc
            inputs, sgbc = self._swish_gated_block('SGB_3_up', inputs, 384, cat=swish3)
            sgb3 = inputs
            sgbc3 = sgbc
            inputs, sgbc = self._swish_gated_block('SGB_2_up', inputs, 288, cat=swish2)
            sgb2 = inputs
            sgbc2 = sgbc
            inputs, sgbc = self._swish_gated_block('SGB_1_up', inputs, 192, cat=swish1)
            sgb1 = inputs
            sgbc1 = sgbc
            sgb_list = [sgb1,sgb2,sgb3,sgb4,sgb5]
            sgbc_list = [sgbc1,sgbc2,sgbc3,sgbc4,sgbc5]
            acts.append(sgb_list)
            acts.append(sgbc_list)

            conv1_1_up = Conv2DLReLU(filters=96, kernel_size=1, inputs=inputs)
            cu1=conv1_1_up
            conv1_2_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_1_up)
            cu2=conv1_2_up
            conv1_3_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_2_up)
            cu3=conv1_3_up
            conv1_4_up = tf.layers.Conv2D(filters=27, kernel_size=1, activation=None,
                                          padding='same')(conv1_3_up)
            cu4=conv1_4_up
            cu_list = [cu1,cu2,cu3,cu4]
            acts.append(cu_list)

            output = (conv1_4_up + 1.0) / 2.0 * 255.0
            output = tf.transpose(output, perm=[3, 1, 2, 0])
            split_r, split_g, split_b = tf.split(output, num_or_size_splits=3, axis=0)
            output = tf.concat([split_r, split_g, split_b], 3)
            self.images_rgb_fake = output
            self.acts = acts
            


        self.params = tf.trainable_variables(scope='SGRU_MODEL')
        self.saver = tf.saver = tf.train.Saver(self.params, max_to_keep=5)

        if summarize:
            with tf.name_scope('summaries'):
                img_count = int(self.images_rgb_fake.get_shape().as_list()[-1]/3)
                # Add each image
                for i in range(img_count):
                    tf.summary.image('Image_{}'.format(i), self.images_rgb_fake[:,:,:,i*3:(i+1)*3])
                # generate histograms for each variable in our model
                for var in self.params:
                    variable_summaries(var)


    def _swish_gated_block(self, name, inputs, filters, cat=None, conv1x1=True):
        """swish_gated block takes in a input tensor and returns two objects, one of
        which is the concat operation found in the SGB, and the other is the
        output of the last convolutional layer (before maxpool or deconv)

        (Think of a better variable name than cat)
        If the cat list is an empty list, we assume we are in the down part of the
        Unet. Otherwise, we are in the up part.
        """
        with tf.variable_scope(name):
            if conv1x1:
                inputs = Conv2DLReLU(filters=filters, kernel_size=1, inputs=inputs)

            conv1 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=inputs)
            conv2 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=conv1)

            if cat is None:
                sgb_op = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
                swish = tf.layers.MaxPooling2D(pool_size=2, strides=2)(inputs)
                swish = SwishMod(swish)
                concat = [sgb_op, swish]
            else:
                sgb_op = Conv2DTransposeLReLU(filters=filters, strides=2, inputs=conv2)
                swish = Conv2DTransposeLReLU(filters=filters, strides=2, inputs=inputs)
                swish = SwishMod(swish)
                concat = [sgb_op, swish, cat]

            return tf.concat(concat, axis=3), conv2


    def save(self, path):
        """need to figure out whether path variable needs .ckpt in the name"""
        self.saver.save(tf.get_default_session(), path)


    def load(self, path):
        self.saver.restore(tf.get_default_session(), path)
