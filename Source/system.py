__doc__ = """
Class emulating the whole system.
"""


class System(object):
    def __init__(self, sensors, devices):
        self.sensor = sensors
        self.device = devices
