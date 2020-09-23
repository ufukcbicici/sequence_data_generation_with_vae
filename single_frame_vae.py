import numpy as np
import tensorflow as tf
from sequence_vae import SequenceVae


class SingleFrameVae(SequenceVae):
    def __init__(self,
                 dataset,
                 encoder_feature_counts,
                 encoder_kernel_width,
                 encoder_fully_connected_layers,
                 decoder_fully_connected_layers,
                 decoder_feature_counts,
                 decoder_kernel_width,
                 decoder_std,
                 z_sample_count,
                 z_dimension):
        super().__init__(dataset, None, None,
                         None, None, None,
                         None, z_dimension, None,
                         None, None, None, decoder_std, z_sample_count)
        self.encoderFeatureCounts = encoder_feature_counts
        self.encoderKernelWidth = encoder_kernel_width
        self.encoderFullyConnectedLayers = encoder_fully_connected_layers
        self.decoderFullyConnectedLayers = decoder_fully_connected_layers
        self.decoderFeatureCounts = decoder_feature_counts
        self.decoderKernelWidth = decoder_kernel_width
        self.encoderOutput = None
        self.encoderStandardDeviations = None
        self.flattenedInput = None

    def build_encoder(self):
        net = self.currentSequences
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            if self.encoderFeatureCounts is None or self.encoderKernelWidth is None:
                self.flattenedInput = tf.contrib.layers.flatten(net)
                net = self.flattenedInput
                # Fully Connected Layers for q(z|x) parameter generation
                for layer_id, layer_dim in enumerate(self.encoderFullyConnectedLayers):
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            else:
                self.encoderOutput = self.build_convolutional_blocks(
                    entry_input=net,
                    module_name="encoder",
                    feature_counts=self.encoderFeatureCounts,
                    kernel_width=self.encoderKernelWidth,
                    use_activations=[True] * len(self.encoderFeatureCounts),
                    ending="pool")
                net = self.encoderOutput
                net = tf.contrib.layers.flatten(net)
                # Global Average Pooling
                # net_shape = net.get_shape().as_list()
                # net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
                # net_shape = net.get_shape().as_list()
                # net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
                # Fully Connected Layers for q(z|x) parameter generation
                for layer_id, layer_dim in enumerate(self.encoderFullyConnectedLayers):
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            # Means for q(z|x)
            self.encoderMeans = tf.layers.dense(net, units=self.zDimension, name="encoder_mean_transform",
                                                activation=None)
            # Variances for q(z|x)
            self.encoderVariances = tf.layers.dense(net, units=self.zDimension, name="encoder_variance_transform",
                                                    activation=tf.nn.softplus)
            self.encoderStandardDeviations = tf.sqrt(self.encoderVariances)
            # Sample z ~ q(z|x) with the reparametrization trick
            z_shape = tf.concat([tf.shape(self.encoderMeans), [self.zSampleCount]], axis=0)
            self.eps = tf.random_normal(shape=z_shape, mean=0.0, stddev=1.0)
            self.z_ = tf.expand_dims(self.encoderMeans, axis=-1) + \
                      tf.expand_dims(self.encoderStandardDeviations, axis=-1) * self.eps
            # Decoder's KL divergence loss: D_{KL}(q(z|x)||p(z))
            self.build_encoder_loss(means_matrix=self.encoderMeans,
                                    standard_deviations_matrix=self.encoderStandardDeviations)

    def build_encoder_loss(self, means_matrix, standard_deviations_matrix):
        self.zPrior = tf.contrib.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(means_matrix),
            scale_diag=tf.ones_like(standard_deviations_matrix))
        self.q_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(
            loc=means_matrix,
            scale_diag=standard_deviations_matrix)
        self.encoderKLDivergence = tf.distributions.kl_divergence(self.q_z_given_x, self.zPrior, allow_nan_stats=False)
        self.expectedEncoderKLDivergence = tf.reduce_mean(self.encoderKLDivergence)

    def get_decoder_output(self, z_):
        net = z_
        # Fully Connected Layers for p(z|x) parameter generation
        for layer_id, layer_dim in enumerate(self.decoderFullyConnectedLayers):
            net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
        # Generate mean vectors for p(z|x)
        x_dimensionality = np.prod(self.dataset.trainingSamples[0]["sequence_image"].shape)
        net = tf.layers.dense(inputs=net, units=x_dimensionality, activation=None)
        return net

    def build_decoder(self, is_inference):
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            if not is_inference:
                # Prepare std. vector
                # Decode: p(x|z) such that z ~ q(z|x)
                for z_id in range(self.zSampleCount):
                    x_means = self.get_decoder_output(z_=self.z_[..., z_id])
                    x_stds = tf.constant(self.decoderStd[np.newaxis, :])
                    x_stds = tf.tile(x_stds, [tf.shape(x_means)[0], 1])
                    self.build_decoder_loss(x_means=x_means, x_stds=x_stds)
                self.expectedDecoderLogProb = tf.reduce_mean(tf.stack(self.logProb_p_x_given_z_list, axis=1))
            else:
                # Sample: z ~ N(z|0,I) and generate
                z_shape = tf.concat([[1], self.encoderMeans.get_shape().as_list()[1:]], axis=0)
                z_normal = tf.random_normal(shape=z_shape, mean=0.0, stddev=1.0)
                self.xMeanInference = self.get_decoder_output(z_=z_normal)

    def build_decoder_loss(self, x_means, x_stds):
        assert x_means.get_shape() == x_stds.get_shape()
        p_x_given_z = tf.contrib.distributions.MultivariateNormalDiag(
            loc=x_means,
            scale_diag=x_stds)
        self.p_x_given_z_list.append(p_x_given_z)
        self.logProb_p_x_given_z_list.append(p_x_given_z.log_prob(value=self.flattenedInput))

    def train(self, batch_size, max_iterations=3000, report_period=25000):
        # Initialize
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=1000000)
        sess.run(tf.global_variables_initializer())
        losses = []
        for iteration_id in range(max_iterations):
            history_tensor, sequence_tensor = self.dataset.get_minibatch(batch_size=batch_size)
            feed_dict = {self.currentSequences: np.expand_dims(sequence_tensor, axis=-1),
                         self.isTraining: True}
            run_ops = [self.optimizer, self.totalVaeLoss]
            results = sess.run(run_ops, feed_dict=feed_dict)
            losses.append(results[1])
            if iteration_id % 10 == 0:
                avg_loss = np.mean(np.array(losses))
                losses = []
                print("Iteration:{0} Avg Loss:{1}".format(iteration_id, avg_loss))
        saver.save(sess, SequenceVae.checkpoint_path)

    def sample_sequence(self, sess, history_sequence, number_of_frames, std_scale=0.1):
        pass

    def sample_signal(self, sess, std_scale=0.1):
        feed_dict = {self.isTraining: False}
        results = sess.run([self.xMeanInference], feed_dict=feed_dict)
        signal = np.reshape(np.squeeze(results[0]), newshape=self.dataset.trainingSamples[0]["sequence_image"].shape)
        return signal

