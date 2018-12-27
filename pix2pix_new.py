from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import time
from datetime import timedelta
import utils

#os.environ["CUDA_VISIBLE_DEVICES"]= "2"

parser = argparse.ArgumentParser()
parser.add_argument("--image_hegiht", type=int, default=128, help="hegiht of the image")
parser.add_argument("--image_width", type=int, default=128, help="width of the image")
parser.add_argument("--checkpoint_dir", default="./checkpoint_dir_train17_L2_GAN", help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--is_test", type=bool, default=False, help="testing of not")
parser.add_argument("--test_image_path", type=str, default="./15_test_mask/", help="path of the image")

parser.add_argument("--train_image_output", type=str, default="./train_L2_GAN_15", help="save_path of the image")

parser.add_argument("--image_path", default="./combined_17/", help="path to folder containing images")
parser.add_argument("--max_epochs", type=int, default=501, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="number of images in batch")
parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")

parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=120.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1, help="weight on GAN term for generator gradient")
parser.add_argument("--is_L2", type=bool, default=True, help="using L1 or L2")
a = parser.parse_args()

EPS = 1e-12


class Pix2pix:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, a.image_hegiht, a.image_width, 1])
        self.target = tf.placeholder(tf.float32, [None, a.image_hegiht, a.image_width, 1])
        self.sess = tf.Session()


    def discrim_conv(self, batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


    def gen_conv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if a.separable_conv:
            return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


    def gen_deconv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if a.separable_conv:
            _b, h, w, _c = batch_input.shape
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))



    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 128, 128, in_channels] => [batch, 64, 64, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, a.ngf)
            layers.append(output)

        layer_specs = [
            a.ngf * 2, # encoder_2: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            a.ngf * 4, # encoder_3: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            a.ngf * 8, # encoder_4: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            a.ngf * 8, # encoder_5: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            a.ngf * 8, # encoder_6: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            a.ngf * 8, # encoder_7: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.gen_conv(rectified, out_channels)
                output = self.batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (a.ngf * 8, 0.5),   # decoder_7: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_6: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.0),   # decoder_5: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 4, 0.0),   # decoder_4: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 2, 0.0),   # decoder_3: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf, 0.0),       # decoder_2: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = self.gen_deconv(rectified, out_channels)
                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 64, 64, ngf * 2] => [batch, 128, 128, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]


    def create_model_and_train(self):
        inputs = self.input
        targets = self.target

        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, a.ndf, stride=2)
                rectified = self.lrelu(convolved, 0.2)
                layers.append(rectified)

            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = a.ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = self.batchnorm(convolved)
                    rectified = self.lrelu(normalized, 0.2)
                    layers.append(rectified)

            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.create_generator(inputs, out_channels)

        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                predict_fake = create_discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

            if a.is_L2:
                gen_loss_L1 = tf.reduce_mean(tf.square(targets - outputs))
            else:
                gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))

            if a.gan_weight == 0:
                gen_loss = gen_loss_L1
            else:
                gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            optimizer_d = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

        with tf.name_scope("generator_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            optimizer_g = gen_optim.minimize(gen_loss, var_list=gen_tvars)

        self.sess.run(tf.global_variables_initializer())

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
            print("parameter_count =", self.sess.run(parameter_count))

        ckpt = tf.train.latest_checkpoint(a.checkpoint_dir)
        saver = tf.train.Saver()
        if ckpt:
            print('syccessfully loaded model from: '+ ckpt)
            saver.restore(self.sess, ckpt)

        image_input, image_gt = utils.read_image(a.image_path)

        if a.is_test:
            images_path = utils.get_all_image_path(a.test_image_path)
            for image_path in images_path:
                image  = utils.imread(image_path, half_size=64, image_size=128)
                image = image[np.newaxis,:,:,np.newaxis]
                output_image = self.sess.run(outputs, feed_dict = {self.input:image})
                utils.save_image_output(output_image, image_path, 'train')

        for i in range(a.max_epochs):

            print('Train epoch {}:-----'.format(i+1))
            start_time = time.time()
            batch = 1
            for batch_input, batch_ouput in utils.batch_iter(image_input, image_gt, a.batch_size, shuffle=True):

                if np.random.randint(2,size=1)[0] == 1:  # random flip
                    batch_input = np.flip(batch_input, axis=1)
                    batch_ouput = np.flip(batch_ouput, axis=1)

                if np.random.randint(2,size=1)[0] == 1:  # random flip
                    batch_input = np.flip(batch_input, axis=1)
                    batch_ouput = np.flip(batch_ouput, axis=1)

                train_dict = {self.input:batch_input, self.target: batch_ouput}
                if  a.gan_weight == 0:
                    gen_loss_L1_value,_  = self.sess.run([gen_loss_L1, optimizer_g], feed_dict=train_dict)

                    if batch % 10 == 0:
                        print('Train batch {}:-----'.format(batch))
                        print('gen_loss_L1_value {0:.6}'.format(gen_loss_L1_value))
                else:
                    discrim_loss_value, gen_loss_GAN_value, gen_loss_L1_value,_ ,_ = self.sess.run([discrim_loss, gen_loss_GAN, gen_loss_L1, optimizer_d, optimizer_g], feed_dict=train_dict)

                    if batch % 4 == 0:
                        print('Train batch {}:-----'.format(batch))
                        print('discrim_loss {0:.6} gen_loss_GAN_value {1:.6} gen_loss_L1_value {2:.6}'.format(discrim_loss_value, gen_loss_GAN_value, gen_loss_L1_value))

                batch += 1

            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

            if i % 100 == 0:
                output_image = self.sess.run(outputs, feed_dict = {self.input:batch_input})
                utils.save_image(batch_input, output_image, i)
                saver.save(self.sess, os.path.join(a.checkpoint_dir, 'step_'+str(i)))

if __name__ == '__main__':
    Model = Pix2pix()
    Model.create_model_and_train()
