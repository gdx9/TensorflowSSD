import numpy as np

image_size = 300
data_dir = "ssd_dataset/"

num_classes = 7
output_channels = num_classes + 1 + 4

output_dir = "output_files"
h5_model_name = "ssd_model.h5"
pb_model_name = "model_ssd_pb.pb"

box_sizes = np.array([
    [[50,65],  [45,60],  [41,67], [60,56]],
    [[111,143],[94,125], [67,151],[114,85]],
    [[87,123], [178,212], [109,143], [67,242]],
    [[181,237],[125,165],[85,236],[173,111]]
    ],dtype=np.float32)
