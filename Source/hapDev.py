__doc__ = """
Class emulating the HapTech device. 
"""
import numpy as np
import time


class HapDev(object):
    def __init__(self, buffer_size, network_delay, window, horizon):
        """
        Initialisation

        :param buffer_size: int, how many samples will be held in the buffer
        :param network_delay: float, the network delay in the device
        :param window: int, how many prior samples to base the prediction on
        :param horizon: int, the prediction horizon
        """
        self.buffer_size = buffer_size
        self.network_delay = network_delay
        self.window = window
        self.horizon = horizon
        self.model = None
        self.sensor_list = []
        self.buffer_list = []
        self.predicted_data = []

    def receive_model(self, model):
        """
        Receive model from source

        :param model: keras model
        """
        self.model = model

    def add_sensor(self, sensor):
        """
        Add sensor to the device

        :param sensor: sensor class
        :return:
        """
        self.sensor_list.append(sensor)

    def remove_sensor(self, sensor_id):
        """
        Remove sensor from the device

        :param sensor_id: int, id of the sensor to be removed
        :return:
        """
        self.sensor_list = [x for x in self.sensor_list if x.id != sensor_id]

    def receive_data(self, sensor, source):
        """
        Receive sample from a sensor

        :param sensor: int, sensor id
        :param source: str, receive 'sensor' or 'label'
        :return: np array, single sample
        """
        if source == 'sample':
            sample = sensor.send_sample()
        elif source == 'label':
            sample = sensor.send_label(self.window)

        # print("Received sample number {sample_nr}".format(sample_nr=sensor.sample_counter))

        return sample

    def predict_values(self, data):
        """
        Predicts the next #horizon samples based on #window past buffers

        :param data: buffered data
        :return: the predicted values as an array
        """
        # First prepare the buffer data for prediction
        # Reshape input pattern to [samples, timesteps, features]
        x = data.reshape(1, len(data), -1)

        # Make prediction
        prediction = self.model.predict(x)

        # Turn to array
        prediction = [x for x in prediction[0, :]]

        return prediction

    def compare_data(self, buffer_labels, predictions):
        """
        Compare the predicted values to the true labels stored in the buffer

        :param buffer_labels:
        :param predictions:
        :return: accuracy & predicted label
        """
        predictions = np.argmax(predictions)
        accuracy = 1

        true_label = buffer_labels[0]
        predicted = predictions

        if true_label != predicted:
            accuracy = 0

        return accuracy, predicted

    def add_to_buffer(self, sensor, length, source):
        """
        Populate the buffer with samples

        :param sensor: int, which sensor to use
        :param length:
        :param source:
        :return: np array of #buffer_size length
        """
        sample_buffer = []

        for i in range(length):
            sample = self.receive_data(sensor, source=source)
            sample_buffer.append(sample)

        if source == 'sample':
            sensor.sample_counter = sensor.sample_counter - (self.window - 1)
        return np.asarray(sample_buffer)

    def run_one_cycle(self, sensor_id):
        """

        :param sensor_id:
        :return:
        """
        # Get the sensor
        sensor = [x for x in self.sensor_list if x.id == sensor_id][0]

        start = time.time()

        # First step is to fill the buffer with samples
        sample_start = time.time()
        sample_buffer = self.add_to_buffer(sensor, self.window, 'sample')
        sample_end = time.time()

        label_start = time.time()
        label_buffer = self.add_to_buffer(sensor, self.horizon, 'label')
        label_end = time.time()

        # Make prediction
        prediction_start = time.time()
        predictions = self.predict_values(sample_buffer)
        prediction_end = time.time()

        # Evaluate the predictions
        evaluation_start = time.time()
        errors_list, predicted_label = self.compare_data(label_buffer, predictions)
        evaluation_end = time.time()

        debug_times = {'Gathering samples': sample_end - sample_start,
                       'Gathering labels': label_end - label_start,
                       'Predictions': prediction_end - prediction_start,
                       'Evaluation': evaluation_end - evaluation_start}


        end = time.time()
        run_time = (end - start) * 1000

        # Print the error values, the average error and how many samples would have been sent
        # print("The error values for each sample are: {errors_list}".format(errors_list=errors_list))
        # print("Cycle runtime is {time}ms".format(time=run_time))
        # print("\n")

        return errors_list, predicted_label, label_buffer[0], run_time, debug_times
