import numpy as np

from data_reader import DataReader
from sequence_vae import SequenceVae

if __name__ == "__main__":
    dataset = DataReader(feature_columns=["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"])
    dataset.prepare_training_set(window_length=500, history_length=500)
    sequence_vae = SequenceVae(dataset=dataset,
                               history_conv_encoder_feature_counts=[16, 32],
                               history_encoder_kernel_width=5,
                               sequence_conv_encoder_feature_counts=[32, 64],
                               sequence_kernel_width=5,
                               bottleneck_feature_counts=[128, 128],
                               bottleneck_kernel_width=3,
                               z_dimension=3,
                               decoder_std=0.5,
                               history_conv_decoder_feature_counts=[16, 32],
                               history_decoder_kernel_width=5,
                               decoder_conv_feature_counts=[32, 32, 32],
                               decoder_kernel_width=5)
    sequence_vae.build_network()
    sequence_vae.train(batch_size=256)
