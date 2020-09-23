import numpy as np
import tensorflow as tf
from data_reader import DataReader
from sequence_vae import SequenceVae
from single_frame_vae import SingleFrameVae

if __name__ == "__main__":
    dataset = DataReader(feature_columns=["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"])
    dataset.prepare_training_set(window_length=500, history_length=500)
    # sequence_vae = SequenceVae(dataset=dataset,
    #                            history_conv_encoder_feature_counts=[16, 32],
    #                            history_encoder_kernel_width=5,
    #                            sequence_conv_encoder_feature_counts=[32, 64],
    #                            sequence_kernel_width=5,
    #                            bottleneck_feature_counts=[128, 128],
    #                            bottleneck_kernel_width=3,
    #                            z_dimension=3,
    #                            decoder_std=dataset.stdArr,
    #                            history_conv_decoder_feature_counts=[16, 32],
    #                            history_decoder_kernel_width=5,
    #                            decoder_conv_feature_counts=[32, 32, 32],
    #                            decoder_kernel_width=5)
    # sequence_vae.build_network()
    sf_vae = SingleFrameVae(dataset=dataset,
                            encoder_feature_counts=None,
                            encoder_kernel_width=None,
                            encoder_fully_connected_layers=[256, 128],
                            decoder_fully_connected_layers=[128, 256],
                            decoder_feature_counts=None,
                            decoder_kernel_width=None,
                            z_sample_count=10,
                            z_dimension=64,
                            decoder_std=
                            np.concatenate([dataset.stdDict[feature_name] for feature_name in dataset.featureColumns]))
    sf_vae.build_network()
    # sf_vae.train(batch_size=32, max_iterations=10000)



    # sess = tf.Session()
    # sequence_vae.load_model(sess)
    # history_tensor, sequence_tensor = dataset.get_minibatch(batch_size=1)
    # dataset.show_data(sequence=sequence_tensor)
    # for idx in range(10):
    #     synthetic_sequence = sequence_vae.sample_sequence(sess=sess,
    #                                                       history_sequence=history_tensor,
    #                                                       number_of_frames=5)
    #     for idj in range(5):
    #         dataset.show_data(sequence=synthetic_sequence[:, 500 * idj: 500 * (idj + 1)])
    # print("X")
