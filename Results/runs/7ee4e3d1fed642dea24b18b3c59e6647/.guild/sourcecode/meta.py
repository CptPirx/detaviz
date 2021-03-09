__doc__ = """
Holds different values common for the project.
"""

# Laptop
# data_path = "C:/Users/au614889/source/repos/Smart-Industry/smart_industry/Data/"

# PC
data_path = "E:/source/datasets/AURSAD/AURSAD.h5"

# Common training parameters
epochs = 100
batch_size = 256
optimizer = 'adam'
learning_rate = 0.001

# Data parameters
dimensionality = 60
window = 100
horizon = 1

# Simulation parameters
network_delay = 1.0
buffer_size = 100
threshold = 0.05
plot_length = 500
