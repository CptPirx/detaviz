__doc__ = """
Class emulating the HapTech device. 
"""

import meta

import numpy as np

import math
import time

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class HapDev(object):
    def __init__(self, buffer_size, network_delay, window, horizon, threshold):
        """
        Initialisation
        :param buffer_size: int, buffer to hold samples
        :param network_delay: float, the network delay in the device
        :param window: int, how many prior samples to base the prediction on
        :param threshold: float, how much the predicted value can differ from the actual measurement
        """
        self.buffer_size = buffer_size
        self.network_delay = network_delay
        self.window = window
        self.horizon = horizon
        self.threshold = threshold
        self.model = None
        self.sensor_list = []
        self.buffer_list = []
        self.predicted_data = []

    def receive_model(self, model):
        """
        Receive model from the cloud
        :param model: keras model
        """
        self.model = model

    def add_sensor(self, sensor):
        """
        Add sensor to the device
        :param sensor:
        :return:
        """
        self.sensor_list.append(sensor)

    def remove_sensor(self, sensor_id):
        """
        Remove sensor from the device
        :param sensor_id:
        :return:
        """
        self.sensor_list = [x for x in self.sensor_list if x.id != sensor_id]

    def receive_data(self, sensor_id):
        """
        Receive sample from a sensor
        :param sensor_id:
        :return: single sample
        """
        sensor = [x for x in self.sensor_list if x.id == sensor_id][0]

        sample, label = sensor.send_sample()

        # print("Received sample number {sample_nr}".format(sample_nr=sensor.sample_counter))

        return sample, label

    def predict_values(self, sensor, data):
        """
        Predicts the next #horizon samples based on #window past buffers
        :param sensor_id: int, which sensor are we predicting
        :return: the predicted values as an array
        """
        # First prepare the buffer data for prediction
        # Reshape input pattern to [samples, timesteps, features]
        # data = np.asarray(self.buffer_list[sensor_id[-self.window:]])
        x = data.reshape(1, len(data), -1)

        # Make prediction
        prediction = self.model.predict(x)

        # Turn to array
        prediction = [x for x in prediction[0, :]]

        return prediction

    def compare_data(self, sensor, buffer, predictions):
        """
        Compare the predicted values to the actual measures stored in the buffer
        :param sensor_id:
        :return:    list of errors
                    average error
                    number of sent samples
        """
        errors = []
        sent_samples = 0

        # Inverse scale the buffer and predictions
        # buffer = sensor.scaler.inverse_transform(buffer.reshape(-1, 1))
        # predictions = sensor.scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1))

        # Calculate individual errors and how many samples would have been sent
        for i in range(len(predictions)):
            actual = buffer[i]
            predicted = predictions[i]
            # rse = math.sqrt((actual - predicted) ** 2)

            # real_error = rse / sensor.sensor_range
            if actual != predicted:
                real_error = True
                errors.append(real_error)
                sent_samples += 1

        # print('Predicted values')
        # print(predictions)
        # print('----')
        # print('Actual values')
        # print(buffer[:len(predictions)])

        # Calculate mean error
        # mean_error = math.sqrt(mean_squared_error(buffer[:len(predictions)], predictions)) / sensor.sensor_range
        mean_error = 0

        return errors, mean_error, sent_samples

    def add_to_buffer(self, sensor_id, length):
        """
        Populate the buffer with samples
        :param sensor_id: int, which sensor to use
        :return: array of #buffer_size length
        """
        stop = False
        sample_buffer = []
        label_buffer = []
        for i in range(length):
            sample, label = self.receive_data(sensor_id)
            sample_buffer.append(sample)
            label_buffer.append(label)

        return np.asarray(sample_buffer), np.asarray(label_buffer)

    def run_one_cycle(self, sensor_id):
        """

        :param sensor_id:
        :return:
        """
        # Get the sensor
        sensor = [x for x in self.sensor_list if x.id == sensor_id][0]

        start = time.time()

        # First step is to fill the buffer with samples
        sample_buffer, label_buffer = self.add_to_buffer(sensor_id, self.window)

        # Make prediction
        predictions = self.predict_values(sensor, sample_buffer)

        # Now fill buffer again with new samples
        sample_buffer, label_buffer = self.add_to_buffer(sensor_id, self.window)

        # Evaluate the predictions
        errors_list, mean_error, sent_samples = self.compare_data(sensor, label_buffer, predictions)

        end = time.time()
        run_time = (end - start) * 1000

        # Print the error values, the average error and how many samples would have been sent
        print("The error values for each sample are: {errors_list}".format(errors_list=errors_list))
        print("The mean error for this cycle is {mean_error}%".format(mean_error=mean_error*100))
        print("The number of samples that would have been sent with threshold value of {threshold}% "
              "is {sent_samples}".format(threshold=self.threshold*100, sent_samples=sent_samples))
        print("Cycle runtime is {time}ms".format(time=run_time))
        print("\n")

        return mean_error, sent_samples, run_time











