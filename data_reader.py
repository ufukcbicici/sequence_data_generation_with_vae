import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import os
from tensorflow.python.platform import gfile


class DataReader:
    def __init__(self, feature_columns):
        self.featureColumns = feature_columns
        self.listOfSequences = []
        search_path = os.path.join("binary_data", '*.csv')
        for csv_path in gfile.Glob(search_path):
            df = pd.read_csv(csv_path)
            self.listOfSequences.append(df)
        self.trainingSamples = []

    def prepare_training_set(self, window_length, history_length):
        file_name = "training_window{0}_history{1}.sav".format(window_length, history_length)
        if os.path.isfile(file_name):
            training_file = open(file_name, "rb")
            self.trainingSamples = pickle.load(training_file)
            training_file.close()
        else:
            feature_count = len(self.featureColumns)
            self.trainingSamples = []
            for sequence_id, sequence in enumerate(self.listOfSequences):
                curr_index = 0
                print("sequence_id={0}".format(sequence_id))
                while curr_index <= sequence.shape[0]:
                    history_image = np.zeros(shape=(feature_count, history_length), dtype=np.float32)
                    sequence_image = np.zeros(shape=(feature_count, window_length), dtype=np.float32)
                    curr_sequence = \
                        sequence[self.featureColumns].iloc[curr_index: min(curr_index + window_length, sequence.shape[0])]
                    history = sequence[self.featureColumns].iloc[max(0, curr_index - history_length): curr_index]
                    sequence_image[:, 0: curr_sequence.T.shape[1]] = curr_sequence.T
                    history_image[:, history_length - history.T.shape[1]:] = history.T
                    self.trainingSamples.append({"history_image": history_image, "sequence_image": sequence_image})
                    curr_index += window_length
            self.trainingSamples = np.array(self.trainingSamples)
            training_file = open(file_name, "wb")
            pickle.dump(self.trainingSamples, training_file)
            training_file.close()

    def get_minibatch(self, batch_size):
        samples = np.random.choice(self.trainingSamples, size=batch_size)
        history_tensor = np.stack([dct["history_image"] for dct in samples], axis=0)
        sequence_tensor = np.stack([dct["sequence_image"] for dct in samples], axis=0)
        return history_tensor, sequence_tensor