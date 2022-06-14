import os
import cv2
import numpy as np
import time

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import sys
sys.path.append('..')

from ssd_settings import *

class ModelPbConverter:
    def __init__(self,):
        pass

    def getFrozenGraph(self, model):
        # get model TF graph
        model_graph = tf.function(lambda x: model(x))

        # get concrete function
        model_graph = model_graph.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        frozen_function = convert_variables_to_constants_v2(model_graph)

        return frozen_function.graph

    def savePbModel(self, frozen_graph, save_dir, model_name):
        # save full tf model
        tf.io.write_graph(graph_or_graph_def=frozen_graph,
                          logdir=save_dir,
                          name=model_name,
                          as_text=False)
        print('model saved')

    def testProtobufModel(self,pb_model_path, test_image_path, num_executions=1, save_output=False):
        # load blob model
        opencv_model = cv2.dnn.readNetFromTensorflow(pb_model_path)

        test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

        input_blob = cv2.dnn.blobFromImage(
            image=test_image,
            scalefactor=1./255.0,
            size=(image_size,image_size)# new size
        )
        print("blob shape: ", input_blob.shape, "with max value of", np.max(input_blob.flatten()))

        exec_times = list()

        for i in range(num_executions):

            start = time.time()# measure time

            opencv_model.setInput(input_blob)
            out = opencv_model.forward()

            end = time.time()# measure time
            seconds = end - start
            exec_times.append(seconds*1000)

            if i % 10 == 0:
                print("iteration:", i)


        print("output shape:", out.shape)
        print("first values: ", out[0,:10])
        print("maxval:", np.max(out.flatten()))
        print("argmax:", np.argmax(out.flatten()))
        print("average execution time:", np.average(exec_times))

        if save_output:
            # write to file
            out_file_path = os.path.join(output_dir, "output.bin")
            newFile = open(out_file_path, "wb")
            newFile.write(out.tobytes())
            print("output.bin saved to", out_file_path)
