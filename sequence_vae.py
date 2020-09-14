import tensorflow as tf
import numpy as np


class SequenceVae:
    def __init__(self, dataset,
                 history_conv_encoder_feature_counts,
                 history_encoder_kernel_width,
                 sequence_conv_encoder_feature_counts,
                 sequence_kernel_width,
                 bottleneck_feature_counts,
                 bottleneck_kernel_width,
                 z_dimension,
                 history_conv_decoder_feature_counts,
                 history_decoder_kernel_width,
                 decoder_conv_feature_counts,
                 decoder_kernel_width,
                 decoder_std,
                 z_sample_count=10):
        self.dataset = dataset
        self.historyConvEncoderFeatureCounts = history_conv_encoder_feature_counts
        self.historyEncoderKernelWidth = history_encoder_kernel_width
        self.sequenceConvEncoderFeatureCounts = sequence_conv_encoder_feature_counts
        self.sequenceKernelWidth = sequence_kernel_width
        self.bottleneckFeatureCounts = bottleneck_feature_counts
        self.bottleneckKernelWidth = bottleneck_kernel_width
        self.decoderConvFeatureCounts = decoder_conv_feature_counts
        self.decoder_kernel_width = decoder_kernel_width
        self.decoderStd = decoder_std
        self.zSampleCount = z_sample_count
        self.zDimension = z_dimension
        self.historyConvDecoderFeatureCounts = history_conv_decoder_feature_counts
        self.historyDecoderKernelWidth = history_decoder_kernel_width
        sequence_shape = [None]
        sequence_shape.extend(list(self.dataset.trainingSamples[0]["sequence_image"].shape))
        sequence_shape.append(1)
        history_shape = [None]
        history_shape.extend(list(self.dataset.trainingSamples[0]["history_image"].shape))
        history_shape.append(1)
        self.currentSequences = tf.placeholder(name="currentSequences", shape=sequence_shape, dtype=tf.float32)
        self.historySequences = tf.placeholder(name="historySequences", shape=history_shape, dtype=tf.float32)
        self.isTraining = tf.placeholder(name="isTraining", dtype=tf.bool)
        self.historyEncoderInput = None
        self.historyDecoderInput = None
        self.historyDecoderDimReduced = None
        self.sequenceEncoderInput = None
        self.bottleneckInput = None
        self.bottleneckOutput = None
        self.upSamplingInput = None
        self.upSamplingOutput = None
        self.encoderParamsBeforePooling = None
        self.encoderParameters = None
        self.encoderMeans = None
        self.encoderVariances = None
        self.encoderMeansConcat = None
        self.encoderVariancesConcat = None
        self.eps = None
        self.z_ = None
        self.xMeans_ = []
        self.xMeanInference = None
        self.zPrior = None
        self.q_z_given_x = None
        self.encoderKLDivergence = None
        self.expectedEncoderKLDivergence = None
        self.p_x_given_z_list = []
        self.logProb_p_x_given_z_list = []
        self.expectedDecoderLogProb = None
        self.totalVaeLoss = None
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = None
        # self.prob_p_x_given_z_list_1 = []
        # self.prob_p_x_given_z_list_2 = []
        # self.decoderLogProb_1 = None
        # self.decoderLogProb_2 = None
        # self.expectedDecoderLogProb_1 = None
        # self.expectedDecoderLogProb_2 = None
        # self.klDivergence_2 = None
        # self.grad_1 = None
        # self.grad_2 = None
        # self.mu = tf.constant([1.2, 0.4, -0.7], dtype=tf.float32)
        # self.sigma = tf.constant([1.9, 1.2, 0.4], dtype=tf.float32)

    def build_convolutional_blocks(self, entry_input, module_name, feature_counts, kernel_width,
                                   use_activations, ending="pool"):
        net = entry_input
        # Encode with convolutional layers
        for layer_id, feature_count in enumerate(feature_counts):
            with tf.variable_scope("{0}_{1}".format(module_name, layer_id)):
                in_filters = net.get_shape().as_list()[-1]
                out_filters = feature_count
                filter_height = 2 * (net.get_shape().as_list()[1] - 1) + 1
                filter_width = 2 * kernel_width + 1
                kernel = [filter_height, filter_width, in_filters, out_filters]
                strides = [1, 1, 1, 1]
                W = tf.get_variable("conv_layer_kernel_{0}".format(layer_id, ), kernel, trainable=True)
                b = tf.get_variable("conv_layer_bias_{0}".format(layer_id), [kernel[-1]], trainable=True)
                net = tf.nn.conv2d(net, W, strides, padding='SAME')
                net = tf.nn.bias_add(net, b)
                if use_activations[layer_id]:
                    net = tf.nn.relu(net)
        if ending == "pool":
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        elif ending == "up_sample":
            up_sampling_in_filters = net.get_shape().as_list()[-1]
            net = tf.layers.conv2d_transpose(
                net,
                filters=int(up_sampling_in_filters / 2),
                kernel_size=2,
                strides=2,
                name="up_sample")
        elif ending == "identity":
            net = tf.identity(net)
        else:
            raise ValueError("Unknown layer ending:{0}".format(ending))
        return net

    def build_encoder(self):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            self.historyEncoderInput = self.build_convolutional_blocks(
                entry_input=self.historySequences,
                module_name="history",
                feature_counts=self.historyConvEncoderFeatureCounts,
                kernel_width=self.historyEncoderKernelWidth,
                use_activations=[True] * len(self.historyConvEncoderFeatureCounts),
                ending="pool")
            self.sequenceEncoderInput = self.build_convolutional_blocks(
                entry_input=self.currentSequences,
                module_name="sequence",
                feature_counts=self.sequenceConvEncoderFeatureCounts,
                kernel_width=self.sequenceKernelWidth,
                use_activations=[True] * len(self.sequenceConvEncoderFeatureCounts),
                ending="pool")
            # The bottleneck layers
            self.bottleneckInput = tf.concat([self.sequenceEncoderInput, self.historyEncoderInput], axis=-1)
            self.bottleneckInput = tf.layers.batch_normalization(inputs=self.bottleneckInput, training=self.isTraining)
            self.bottleneckOutput = self.build_convolutional_blocks(
                entry_input=self.bottleneckInput,
                module_name="bottleneck",
                feature_counts=self.bottleneckFeatureCounts,
                kernel_width=self.bottleneckKernelWidth,
                use_activations=[True] * len(self.bottleneckFeatureCounts),
                ending="identity")
            # Up sampling layers
            self.upSamplingOutput = self.build_convolutional_blocks(
                entry_input=self.bottleneckOutput,
                module_name="up_sample",
                feature_counts=[self.bottleneckOutput.get_shape().as_list()[-1]],
                kernel_width=self.bottleneckKernelWidth,
                use_activations=[False],
                ending="up_sample")
            # Calculate outputs
            net = self.upSamplingOutput
            output_filter_height = int(len(self.dataset.featureColumns) / self.zDimension)
            with tf.variable_scope("output_calculation"):
                in_filters = net.get_shape().as_list()[-1]
                out_filters = 2
                kernel = [1, 1, in_filters, out_filters]
                strides = [1, 1, 1, 1]  # [batch, height, width, channels]
                W = tf.get_variable("output_calculation_layer_kernel", kernel, trainable=True)
                b = tf.get_variable("output_calculation_layer_bias", [kernel[-1]], trainable=True)
                net = tf.nn.conv2d(net, W, strides, padding='SAME')
                self.encoderParamsBeforePooling = tf.nn.bias_add(net, b)
                self.encoderParameters = tf.nn.max_pool(self.encoderParamsBeforePooling,
                                                        ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
            self.encoderMeans, self.encoderVariances = tf.split(self.encoderParameters, num_or_size_splits=2, axis=-1)
            # self.encoderMeans = tf.squeeze(self.encoderMeans)
            # self.encoderVariances = tf.squeeze(self.encoderVariances)
            # # Variances must always be non-negative
            self.encoderVariances = tf.nn.softplus(self.encoderVariances)
            # Sample z ~ q(z|x) with the reparametrization trick
            z_shape = tf.concat([tf.shape(self.encoderVariances)[:-1], [self.zSampleCount]], axis=0)
            self.eps = tf.random_normal(shape=z_shape, mean=0.0, stddev=1.0)
            self.z_ = self.encoderMeans + tf.sqrt(self.encoderVariances) * self.eps
            self.encoderMeansConcat = self.convert_to_matrix(tensor_=self.encoderMeans)
            self.encoderVariancesConcat = self.convert_to_matrix(tensor_=self.encoderVariances)
            # Decoder's KL divergence loss: D_{KL}(q(z|x)||p(z))
            self.zPrior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(self.encoderMeansConcat),
                scale_diag=tf.ones_like(self.encoderVariancesConcat))
            self.q_z_given_x = tf.contrib.distributions.MultivariateNormalDiag(
                loc=self.encoderMeansConcat,
                scale_diag=self.encoderVariancesConcat)
            self.encoderKLDivergence = tf.distributions.kl_divergence(self.q_z_given_x, self.zPrior,
                                                                      allow_nan_stats=False)
            self.expectedEncoderKLDivergence = tf.reduce_mean(self.encoderKLDivergence)

    def get_decoder_output(self, z_):
        z_with_history = tf.concat([z_, self.historyDecoderDimReduced], axis=-1)
        decoder_output = self.build_convolutional_blocks(
            entry_input=z_with_history,
            module_name="decoder",
            feature_counts=self.decoderConvFeatureCounts,
            kernel_width=self.decoder_kernel_width,
            use_activations=[_i != len(self.decoderConvFeatureCounts) - 1 for _i in
                             range(len(self.decoderConvFeatureCounts))],
            ending="identity")
        with tf.variable_scope("x_mean_generator"):
            # Reduce to a single feature map
            in_filters = decoder_output.get_shape().as_list()[-1]
            out_filters = 1
            kernel = [1, 1, in_filters, out_filters]
            strides = [1, 1, 1, 1]
            W = tf.get_variable("x_mean_generator_kernel", kernel, trainable=True)
            b = tf.get_variable("x_mean_generator_bias", [kernel[-1]], trainable=True)
            net = tf.nn.conv2d(decoder_output, W, strides, padding='SAME')
            net = tf.nn.bias_add(net, b)
            # Up sample to original size
            # up_sampling_in_filters = net.get_shape().as_list()[-1]
            net = tf.layers.conv2d_transpose(
                net,
                filters=1,
                kernel_size=[2, 1],
                strides=[2, 1],
                name="x_mean_generator_up_sample")
            x_means = self.convert_to_matrix(tensor_=net)
            return x_means

    def build_decoder(self, z_input, sample_count, is_inference):
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            self.historyEncoderInput = self.build_convolutional_blocks(
                entry_input=self.historySequences,
                module_name="history",
                feature_counts=self.historyConvEncoderFeatureCounts,
                kernel_width=self.historyEncoderKernelWidth,
                use_activations=[_i != len(self.historyConvEncoderFeatureCounts) - 1 for _i in
                                 range(len(self.historyConvEncoderFeatureCounts))],
                ending="identity")
            # Reduce history dimension to single feature
            with tf.variable_scope("history_dimension_reduction"):
                in_filters = self.historyEncoderInput.get_shape().as_list()[-1]
                out_filters = 1
                kernel = [1, 1, in_filters, out_filters]
                strides = [1, 1, 1, 1]  # [batch, height, width, channels]
                W = tf.get_variable("history_dim_reduction_kernel", kernel, trainable=True)
                b = tf.get_variable("history_dim_reduction_bias", [kernel[-1]], trainable=True)
                net = tf.nn.conv2d(self.historyEncoderInput, W, strides, padding='SAME')
                self.historyDecoderDimReduced = tf.nn.bias_add(net, b)
                self.historyDecoderDimReduced = tf.nn.max_pool(self.historyDecoderDimReduced, ksize=[1, 2, 1, 1],
                                                               strides=[1, 2, 1, 1], padding='SAME')
                self.historyDecoderDimReduced = tf.layers.batch_normalization(
                    inputs=self.historyDecoderDimReduced, training=self.isTraining)
            if not is_inference:
                # Decode: p(x|z,history) such that z ~ q(z|x,history)
                self.xMeans_ = []
                for z_id in range(sample_count):
                    z_ = tf.expand_dims(z_input[..., z_id], axis=-1)
                    x_means = self.get_decoder_output(z_=z_)
                    self.xMeans_.append(x_means)
                    p_x_given_z = tf.contrib.distributions.MultivariateNormalDiag(
                        loc=x_means,
                        scale_diag=self.decoderStd * tf.ones_like(x_means))
                    self.p_x_given_z_list.append(p_x_given_z)
                    X_ = self.convert_to_matrix(tensor_=self.currentSequences)
                    self.logProb_p_x_given_z_list.append(p_x_given_z.log_prob(value=X_))
                self.expectedDecoderLogProb = tf.reduce_mean(tf.stack(self.logProb_p_x_given_z_list, axis=1))
            else:
                # Sample: z ~ N(z|0,I) and generate
                z_shape = tf.concat([[1], tf.shape(self.currentSequences)[1:2], [1]], axis=0)
                z_normal = tf.random_normal(shape=z_shape, mean=0.0, stddev=1.0)
                self.xMeanInference = self.get_decoder_output(z_=z_normal)

    def build_network(self):
        # ****** Encoder: q(z|x) ******
        self.build_encoder()
        # ****** Encoder: q(z|x) ******

        # ****** Decoder: p(x|z) ******
        self.build_decoder(z_input=self.z_, sample_count=self.zSampleCount, is_inference=False)
        # ****** Decoder: p(x|z) ******

        # ****** Total Vae Loss ******
        self.totalVaeLoss = self.expectedEncoderKLDivergence - self.expectedDecoderLogProb
        self.optimizer = tf.train.AdamOptimizer().minimize(self.totalVaeLoss, global_step=self.globalStep)
        # ****** Total Vae Loss ******

    def convert_to_matrix(self, tensor_):
        mt = tf.squeeze(tensor_)
        mt = tf.transpose(mt, perm=[0, 2, 1])
        m_shape = tf.shape(mt)
        mt = tf.reshape(mt, shape=(m_shape[0] * m_shape[1], m_shape[2]))
        return mt

    def train(self, batch_size, max_iterations=2500, report_period=25000):
        # Initialize
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=1000000)
        sess.run(tf.global_variables_initializer())
        losses = []
        for iteration_id in range(max_iterations):
            history_tensor, sequence_tensor = self.dataset.get_minibatch(batch_size=batch_size)
            feed_dict = {self.currentSequences: np.expand_dims(sequence_tensor, axis=-1),
                         self.historySequences: np.expand_dims(history_tensor, axis=-1)}
            run_ops = [self.optimizer, self.totalVaeLoss]
            results = sess.run(run_ops, feed_dict=feed_dict)
            losses.append(results[1])
            if iteration_id % 10 == 0:
                avg_loss = np.mean(np.array(losses))
                losses = []
                print("Iteration:{0} Avg Loss:{1}".format(iteration_id, avg_loss))
        saver.save(sess, "D://sensor_data_generation//models//vae_model.ckpt")

        # # run_ops = [self.encoderParameters,
        # #            self.encoderParamsBeforePooling,
        # #            self.encoderMeans,
        # #            self.encoderVariances,
        # #            self.encoderMeansConcat,
        # #            self.encoderVariancesConcat,
        # #            self.eps,
        # #            self.z_,
        # #            [tf.expand_dims(self.z_[..., z_id], axis=-1) for z_id in range(self.zSampleCount)],
        # #            self.xMeans_,
        # #            self.zPrior.mean(),
        # #            self.q_z_given_x.mean(),
        # #            self.encoderKLDivergence,
        # #            self.expectedEncoderKLDivergence,
        # #            self.decoderLogProb_1,
        # #            self.decoderLogProb_2,
        # #            self.expectedDecoderLogProb_1,
        # #            self.expectedDecoderLogProb_2,
        # #            self.grad_1,
        # #            self.grad_2]
        # # results = sess.run(run_ops, feed_dict=feed_dict)
        # # assert np.array_equal(results[0], np.stack(results[1:], axis=-1))
        # # assert np.allclose(results[1] + np.sqrt(results[2]) * results[3], results[4])
        # # assert all([np.array_equal(results[1][i, :, j, 0], results[3][i * 500 + j, :]) for i in range(128) for j in
        # #             range(500)])
        # # assert all([np.array_equal(results[2][i, :, j, 0], results[4][i * 500 + j, :]) for i in range(128) for j in
        # #             range(500)])
        # print("X")

    def kl_divergence_of_diagonal_gaussians(self, mu_p, sigma_p, mu_q, sigma_q):
        s_p = np.square(sigma_p)
        s_q = np.square(sigma_q)
        Sigma_p = np.diag(s_p)
        Sigma_q = np.diag(s_q)
        # det_p = np.prod(sigma_p)
        # det_q = np.prod(sigma_q)
        det_p = np.linalg.det(Sigma_p)
        det_q = np.linalg.det(Sigma_q)
        A = np.log(det_q / det_p)
        B = mu_p.shape[0]
        C = np.dot(np.dot((mu_p - mu_q).T, np.linalg.inv(Sigma_q)), (mu_p - mu_q))
        D = np.trace(np.dot(np.linalg.inv(Sigma_q), Sigma_p))
        kl_divergence = 0.5 * (A - B + C + D)
        return kl_divergence

    def tf_encoder_kl_divergence_loss(self, mu_p, sigma_p):
        var_p = tf.square(sigma_p)
        log_var_p = tf.log(var_p)
        log_det_p = tf.reduce_sum(log_var_p, axis=1)
        k = tf.cast(tf.shape(mu_p)[1], dtype=tf.float32)
        trace_p = tf.reduce_sum(var_p, axis=1)
        mu_p_dot_mu_p = tf.reduce_sum(tf.multiply(mu_p, mu_p), axis=1)
        kl_divergences = 0.5 * ((mu_p_dot_mu_p + trace_p - log_det_p) - k)
        return kl_divergences

    def tf_multivariate_isotropic_gaussian_log_prob(self, x_, mu_p, sigma):
        var = sigma ** 2.0
        x_minus_mu = x_ - mu_p
        x_minus_mu_dot = tf.multiply(x_minus_mu, x_minus_mu)
        A = tf.exp(-0.5 * (1.0 / var) * tf.reduce_sum(x_minus_mu_dot, axis=1))
        k = tf.cast(tf.shape(x_)[1], dtype=tf.float32)
        B = 1.0 / tf.sqrt(tf.pow(2.0 * np.pi, k) * tf.pow(var, k))
        pdf = A * B
        return tf.log(pdf)
