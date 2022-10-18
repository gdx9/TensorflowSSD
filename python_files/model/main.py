from ssd_model_trainer import *

if __name__ == '__main__':
    """
        train and save model to .h5-format
    """

    box_converter = BoxConverter(box_sizes, image_size, num_classes, data_dir, is_save_=False)
    #box_converter.show_layer_boxes()
    trainer = SsdModelTrainer(data_dir, box_converter)

    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    trainer.train(learning_rates, 80)

    ## test results
    trainer.showTestResults(is_dataset_test=True)

    # save
    # centers.bin
    # wh.bin
    box_converter.saveBoxData()

    # convert model to supported format
    supported_model = trainer.convertToSupportedFormat()
    # save model's .h5-file
    supported_model.save(os.path.join(output_dir, h5_model_name))
