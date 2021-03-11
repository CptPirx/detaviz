__doc__ = """
Class emulating the sensor. Responsible for loading the data and sending it further.
"""


class Sensor(object):
    def __init__(self, axle_nr, buffer_size, dataset, dataset_labels):
        """
        Initialisation
        :param axle_nr: int, which axle to use
        """
        self.sensor_domain = axle_nr
        self.id = axle_nr
        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.sample_counter = 0
        self.label_counter = 100
        self.buffer_size = buffer_size

    def send_sample(self):
        """
        Send single sample
        :return: float, single sample
        """
        to_send_sample = self.dataset[self.sample_counter]

        self.sample_counter += 1

        return to_send_sample

    def send_label(self, window):
        """
        Send single label
        :return: float, single sample
        """
        to_send_label = self.dataset_labels[self.label_counter]

        self.label_counter += 1

        return to_send_label

    def check_end(self):
        if self.sample_counter > self.dataset.shape[0] - 2 * self.buffer_size:
            return True
