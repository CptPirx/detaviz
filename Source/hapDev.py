__doc__ = """
Class emulating the HapTech device. 
"""
import numpy as np
import time


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

    def receive_data(self, sensor, source):
        """
        Receive sample from a sensor

        :param sensor:
        :return: single sample
        """
        if source == 'sample':
            sample = sensor.send_sample()
        elif source == 'label':
            sample = sensor.send_label(self.window)

        # print("Received sample number {sample_nr}".format(sample_nr=sensor.sample_counter))

        return sample

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

    def compare_data(self, buffer, predictions):
        """
        Compare the predicted values to the actual measures stored in the buffer

        :param sensor_id:
        :return:    list of errors
                    average error
                    number of sent samples
        """
        predictions = np.argmax(predictions)
        errors = 1

        actual = buffer[0]
        predicted = predictions

        if actual != predicted:
            errors = 0

        return errors, predicted

    def add_to_buffer(self, sensor, length, source):
        """
        Populate the buffer with samples

        :param sensor_id: int, which sensor to use
        :return: list of #buffer_size length
        """
        sample_buffer = []

        for i in range(length):
            sample = self.receive_data(sensor, source=source)
            sample_buffer.append(sample)

        if source == 'sample':
            sensor.sample_counter = sensor.sample_counter - (self.window - 1)
        return np.asarray(sample_buffer)

    def run_one_cycle(self, sensor_id, current_cycle):
        """

        :param current_cycle:
        :param sensor_id:
        :return:
        """
        # Get the sensor
        sensor = [x for x in self.sensor_list if x.id == sensor_id][0]

        start = time.time()

        # First step is to fill the buffer with samples
        sample_buffer = self.add_to_buffer(sensor, self.window, 'sample')
        label_buffer = self.add_to_buffer(sensor, self.horizon, 'label')

        # Make prediction
        predictions = self.predict_values(sensor, sample_buffer)

        # Evaluate the predictions
        errors_list, predicted_label = self.compare_data(label_buffer, predictions)

        end = time.time()
        run_time = (end - start) * 1000

        # Print the error values, the average error and how many samples would have been sent
        # print("The error values for each sample are: {errors_list}".format(errors_list=errors_list))
        # print("Cycle runtime is {time}ms".format(time=run_time))
        # print("\n")

        return errors_list, predicted_label, label_buffer[0], run_time











