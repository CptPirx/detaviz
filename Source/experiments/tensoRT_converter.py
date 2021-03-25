from tensorflow.python.compiler.tensorrt import trt_convert as trt

file_path = '/home/blez/Projects/anomaly_simulation/Zoo/Results/runs/27760a9c70ff4042931ceb142a3f651e/model'
save_path = '/home/blez/Projects/anomaly_simulation/Zoo/Results/runs/27760a9c70ff4042931ceb142a3f651e/model_converted'

# Convert the SavedModel
converter = trt.TrtGraphConverterV2(input_saved_model_dir=file_path)
converter.convert()

# Save the converted model
converter.save(save_path)