from model_pb_converter import *

if __name__ == '__main__':
    pb_converter = ModelPbConverter()

    h5_model = tf.keras.models.load_model(os.path.join(output_dir,h5_model_name))
    #h5_model.summary()
    frozen_graph = pb_converter.getFrozenGraph(h5_model)
    pb_converter.savePbModel(frozen_graph, output_dir, pb_model_name)

    # test protobuf model
    pb_converter.testProtobufModel(os.path.join(output_dir, pb_model_name),
                                    os.path.join(data_dir, "0.png"),
                                    150, save_output=True)
