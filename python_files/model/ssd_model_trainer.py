import numpy as np
import numpy.matlib

import cv2
import random
import math

import os
from os import listdir
from os import path
from os.path import isfile, join

import tensorflow as tf
import tensorflow.keras.backend as K#.concatenate

import tensorflow_io as tfio

import sys
sys.path.append('..')

from ssd_settings import *
from ssd_utils import *


# box converter
class BoxConverter:
    def _prepareDefaultBoxes(self):

        max_box_size = 38
        num_boxes = list()
        self.layer_widths = list()
        for sz in self.box_sizes:
            self.layer_widths.append(max_box_size)
            num_boxes.append(len(sz))

            max_box_size = math.ceil(max_box_size/2)

        assert len(self.box_sizes) == len(num_boxes)
        assert len(self.box_sizes) == len(self.layer_widths)

        self.total_boxes_num = sum([lw*lw*nb for lw,nb in zip(self.layer_widths, num_boxes)])

        self.centers      = np.zeros((self.total_boxes_num, 2), dtype=np.float32)
        self.wh           = np.zeros((self.total_boxes_num, 2), dtype=np.float32)
        self.boxes_coords = np.zeros((self.total_boxes_num, 4), dtype=np.float32)
        print("total_boxes_num:", self.total_boxes_num)

        # calculating the default box centers and width,height
        idx = 0
        for grid_size, box_size in zip(self.layer_widths, self.box_sizes):
            step_size = self.image_size * 1.0 / grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    pos = idx + (i*grid_size+j) * len(box_size)

                    # same centers for all aspect ratios
                    self.centers[pos : pos + len(box_size), :] = j*step_size + step_size/2, i*step_size + step_size/2
                    self.wh[pos : pos + len(box_size), :] = box_size

            idx += grid_size * grid_size * len(box_size)

        # (x,y) coordinates of top left and bottom right
        self.boxes_coords[:,0] = self.centers[:,0] - self.wh[:,0]/2
        self.boxes_coords[:,1] = self.centers[:,1] - self.wh[:,1]/2
        self.boxes_coords[:,2] = self.centers[:,0] + self.wh[:,0]/2
        self.boxes_coords[:,3] = self.centers[:,1] + self.wh[:,1]/2

        print("default boxes are ready")

    def _adjustDataForNewSize(self, x, y, width, height, old_size=(640,480), new_size=(image_size,image_size)):
        x = x * new_size[0] // old_size[0]
        y = y * new_size[1] // old_size[1]
        width  = width * new_size[0] // old_size[0]
        height = height * new_size[1] // old_size[1]

        return x,y,width,height

    def _readLabelFileData(self, path):
        label_data = []
        with open(path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                label_data.append([class_label, x, y, width, height])
        return label_data

    def _labelDataToBoxes(self,label_data):
        output = np.zeros((self.total_boxes_num, 1 + 4))# class_number, center(x,y), w, h
        output[:,0] = self.total_classes

        for cls,x,y,width,height in label_data:
            x_coord_0, y_coord_0, width, height = self._adjustDataForNewSize(x,y,width,height)

            # 0
            bbox = np.zeros(4)
            bbox[:2] = [x_coord_0, y_coord_0]
            bbox[2:] = [x_coord_0+width, y_coord_0+height]

            # all default boxes with IoU > threshold
            box_idxs = bestIoU(bbox, self.boxes_coords, self.total_boxes_num, threshold=0.6)
            print("boxes found:", len(box_idxs))

            output[box_idxs,0] = cls
            output[box_idxs,1] = (bbox[0] + bbox[2])/2.0 - self.centers[box_idxs,0]
            output[box_idxs,2] = (bbox[1] + bbox[3])/2.0 - self.centers[box_idxs,1]
            output[box_idxs,3] = width - self.wh[box_idxs,1]
            output[box_idxs,4] = height - self.wh[box_idxs,0]

        return output

    def generateSaveSsdBoxes(self):
        # get image filenames without extension
        indexes = [int(f[:-4]) for f in listdir(self.data_dir) if f[-3:] == 'png']

        for idx in indexes:
            # get label txt-file path
            path = os.path.join(self.data_dir, str(idx) + '.txt')

            label_data = self._readLabelFileData(path)
            generated_boxes = self._labelDataToBoxes(label_data).astype(np.float32)

            # save
            if self.is_save:
                npy_path = os.path.join(self.data_dir, str(idx) + '.npy')
                np.save(npy_path, generated_boxes)

    def _xywh_to_points(self, xywh, box_num):
        """
            calculate and return start point and end point
        """
        box_center_x_y = self.centers[box_num] + xywh[:2]
        box_wh = self.wh[box_num] + xywh[2:]

        x = box_center_x_y[0] - box_wh[0]/2
        y = box_center_x_y[1] - box_wh[1]/2

        start_point = (int(x), int(y))
        end_point   = (int(x + box_wh[0]), int(y + box_wh[1]))

        return start_point, end_point

    def get_best_boxes(self,y_pred, output_channels, threshold=0.8):

        class_predictions = tf.nn.softmax(y_pred[:,:output_channels-4],axis=1)
        box_max_class = np.argmax(class_predictions, axis=-1)# list of classes with highest probability in box

        assert len(box_max_class) == len(y_pred)

        recognized_data = list()

        for box_pos in range(len(class_predictions)):
            # check if class is background
            if num_classes == box_max_class[box_pos]:
                continue
            # get max value
            val = class_predictions[box_pos, box_max_class[box_pos]]
            #print(val)
            if val > threshold:
                start_point, end_point = self._xywh_to_points(y_pred[box_pos, output_channels-4:], box_pos)

                recognized_data.append(RecognitionData(box_max_class[box_pos], val, start_point, end_point))

        return recognized_data

    def get_best_boxes_label(self, y_label):
        box_pos = np.argwhere(y_label[:,0] != num_classes)

        recognized_data = list()

        for pos in box_pos:
            pos = pos[0]
            val = y_label[pos]
            start_point, end_point = self._xywh_to_points(val[-4:], pos)

            recognized_data.append(RecognitionData(int(val[0]), 1.0, start_point, end_point))

        return recognized_data

    def getUniqueBoxes(self, found_boxes):
        unique_boxes = []
        for bx in found_boxes:
            found = False
            for ubx in unique_boxes:
                if ubx.class_num == bx.class_num and ubx.start_point == bx.start_point and ubx.end_point == bx.end_point:
                    found = True
                    break
            if found == False:
                unique_boxes.append(bx)

        return unique_boxes

    def calc_mAP_f1(self, detections, ground_truths, threshold=0.5):
        epsilon = 1e-6

        tp = 0

        for gt in ground_truths:
            for det in detections:
                # for same class
                if det.class_num == gt.class_num:
                    iou = IoU(np.array([[det.start_point[0], det.start_point[1], det.end_point[0], det.end_point[1]]]),
                             np.array([[gt.start_point[0], gt.start_point[1], gt.end_point[0], gt.end_point[1]]]))

                    if iou > threshold:
                        tp += 1
                        break

        recall = tp / (len(detections) + epsilon)
        precision = tp / (len(ground_truths) + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        return f1


    def _saveBinArrayToFile(self, filename, data):
        # write to file
        new_file = open(filename, "wb")
        new_file.write(data.tobytes())

    def saveBoxData(self):
        """
            saves centers and wh arrays to binary files
        """
        self._saveBinArrayToFile(os.path.join(output_dir, "centers.bin"), self.centers)
        self._saveBinArrayToFile(os.path.join(output_dir, "wh.bin"), self.wh)

    def show_layer_boxes(self):
        COLOR_VIOLET = (226,43, 138)
        COLOR_GREEN  = (34, 139,34)
        COLOR_PINK   = (133,21, 199)
        COLOR_ORANGE = (0,  140,255)

        pos = 0

        for grid_size in self.layer_widths:
            print("show grid of", grid_size)

            image = np.zeros([image_size,image_size,3], dtype=np.uint8)

            for x in range(grid_size):
                for y in range(grid_size):
                    center_position = self.centers[pos].astype(np.int32)
                    cv2.circle(image, center_position, 1, (0,0,255), 1)

                    # show boxes in the middle of image
                    if x == grid_size//2 and y == grid_size//2:
                        wh_values0 = (self.wh[pos+0].astype(np.int32)) // 2
                        wh_values1 = (self.wh[pos+1].astype(np.int32)) // 2
                        wh_values2 = (self.wh[pos+2].astype(np.int32)) // 2
                        wh_values3 = (self.wh[pos+3].astype(np.int32)) // 2

                        start_pos0 = center_position - wh_values0
                        end_pos0   = center_position + wh_values0
                        start_pos1 = center_position - wh_values1
                        end_pos1   = center_position + wh_values1
                        start_pos2 = center_position - wh_values2
                        end_pos2   = center_position + wh_values2
                        start_pos3 = center_position - wh_values3
                        end_pos3   = center_position + wh_values3

                        cv2.rectangle(image, start_pos0, end_pos0, COLOR_VIOLET, 1)
                        cv2.rectangle(image, start_pos1, end_pos1, COLOR_GREEN,  1)
                        cv2.rectangle(image, start_pos2, end_pos2, COLOR_PINK,   1)
                        cv2.rectangle(image, start_pos3, end_pos3, COLOR_ORANGE, 1)

                    pos += box_sizes.shape[1]# 4

            cv2.imshow("image", image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def __init__(self, box_sizes_, image_size_, total_classes_, data_dir_, is_save_):

        self.box_sizes = box_sizes_
        self.image_size = image_size_
        self.total_classes = total_classes_
        self.data_dir = data_dir_
        self.is_save = is_save_

        # prepare default boxes: centers, wh, boxes
        self._prepareDefaultBoxes()

        # txt to npy
        self.generateSaveSsdBoxes()

class ShowPredCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs_, check_image_, check_label_, box_converter_, learning_rates_):
        self.epochs = epochs_
        self.check_image = check_image_
        self.check_label = check_label_
        self.box_converter = box_converter_

        self.learning_rates = learning_rates_
        self.lr_pos_step = self.epochs // len(self.learning_rates)
        self.lr_position = 0

    def on_epoch_end(self, epoch, logs={}):
        predicted = self.model.predict(self.check_image)

        recognized_data = self.box_converter.get_best_boxes(predicted[0], output_channels, 0.5)
        recognized_data = np.array(non_max_suppression(recognized_data, 0.3, 0.65))# nms

        """
        # correct labels
        correct_data = self.box_converter.get_best_boxes_label(self.check_label)
        correct_data = self.box_converter.getUniqueBoxes(correct_data)

        # mean average precision
        f1_score = self.box_converter.calc_mAP_f1(recognized_data, correct_data)
        print(" mAP f1 on 1 picture: {:.4f}", f1_score)

        if epoch == 30:
            print("epoch 30")
            for cd in correct_data:
                print("cd:", cd)
            for rd in recognized_data:
                print("rd:", rd)
        """
        # clone image
        image = self.check_image[0].numpy().copy()

        # draw
        draw_all_rectanges(image, recognized_data)
        #draw_all_rectanges(image, correct_data)
        #draw_average_rectangle(image, recognized_data)

        image = cv2.resize(image, (600,600))
        cv2.imshow("check image", image)
        cv2.waitKey(1)

        # change learning rate
        if epoch > 0 and epoch % self.lr_pos_step == 0:
            self.lr_position += 1
            self.model.optimizer.lr.assign(self.learning_rates[self.lr_position])
            print("\n\nlearning rate:", self.model.optimizer.lr.read_value().numpy())

        # for last epoch
        if epoch == (self.epochs - 1):
            cv2.destroyAllWindows()


@tf.function
def prepare_image_tensor(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image,channels=3)
    #img_h, img_w  = image.shape[:2]

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [image_size, image_size])

    # random change of colors
    image = tf.image.adjust_saturation(image, tf.random.uniform([], maxval=5, dtype=tf.float32))
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.2,seed=tf.random.uniform([2], maxval=5, dtype=tf.int32))
    image = tf.image.stateless_random_contrast(
        image, lower=0.5, upper=0.9, seed=tf.random.uniform([2], maxval=5, dtype=tf.int32))

    image = tfio.experimental.color.bgr_to_rgb(image)

    return image

@tf.function
def prepareLabelData(label_path):
    data = tf.io.read_file(label_path)
    data = tf.io.decode_raw(data, tf.float32)
    data = data[32:]
    data = tf.reshape(data, [7720, 5])

    return data

@tf.function
def prepare_image_data(filename):
    img_path = data_dir + tf.strings.as_string(filename) + '.png'
    label_path = data_dir + tf.strings.as_string(filename) + '.npy'

    img = prepare_image_tensor(img_path)
    label = prepareLabelData(label_path)

    return (img,label,)

class SsdModelTrainer:
    def _splitFolderImageNames(self, image_dir_path, train_split_percentage):
        """
            return train, test and validation lists (in INTEGER format) of png-file names
        """

        all_image_pathes = [int(f[:-4]) for f in listdir(image_dir_path) if f[-3:] == 'png']
        random.shuffle(all_image_pathes)

        files_number = len(all_image_pathes)
        train_number = files_number * train_split_percentage // 100
        train_files = all_image_pathes[:train_number]

        test_number = (files_number - train_number) // 2
        test_files = all_image_pathes[train_number:(train_number + test_number)]
        valid_files = all_image_pathes[(train_number + test_number):]

        return train_files, test_files, valid_files

    def _prepareDataset(self,):
        train_files, test_files, valid_files = self._splitFolderImageNames(self.data_dir, train_split_percentage=80)
        len(train_files), len(test_files), len(valid_files)

        BATCH_SIZE = 32
        dataset_train = tf.data.Dataset.from_tensor_slices(train_files).shuffle(64)
        dataset_train = dataset_train.map(prepare_image_data)
        self.dataset_train = dataset_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        dataset_test = tf.data.Dataset.from_tensor_slices(test_files)
        dataset_test = dataset_test.map(prepare_image_data)
        self.dataset_test = dataset_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        dataset_valid = tf.data.Dataset.from_tensor_slices(valid_files)
        dataset_valid = dataset_valid.map(prepare_image_data)
        self.dataset_valid = dataset_valid.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # get image for test
        for check_image, check_label in self.dataset_train.take(1):
            pass
        self.check_image = tf.expand_dims(check_image[0], axis=0)
        self.check_label = check_label[0]
        print(self.check_image.shape, self.check_label.shape)


    def _buildModel(self):
        backbone = tf.keras.applications.MobileNetV2(
                include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet'
            )
        backbone.trainable = False

        for layer in backbone.layers:
            layer.trainable = False
        #keras.utils.plot_model(backbone, "backbone.png", show_shapes=True)

        output0 = backbone.get_layer('block_6_expand_relu')
        output1 = backbone.get_layer('block_13_expand_relu')

        conv0 = tf.keras.layers.Conv2D(256,1,strides=1,padding='same',activation='relu',kernel_initializer='he_normal',name='SSD_11')(output1.output)
        conv1 = tf.keras.layers.Conv2D(512,3,strides=2,padding='same',activation='relu',kernel_initializer='he_normal',name='SSD_12')(conv0)
        conv2 = tf.keras.layers.Conv2D(128,1,strides=1,padding='same',activation='relu',kernel_initializer='he_normal',name='SSD_21')(conv1)
        conv3 = tf.keras.layers.Conv2D(256,3,strides=2,padding='same',activation='relu',kernel_initializer='he_normal',name='SSD_22')(conv2)
        conv4 = tf.keras.layers.Conv2D(128,1,strides=1,padding='same',activation='relu',kernel_initializer='he_normal',name='SSD_31')(conv3)

        DROPOUT_VALUE = 0.5
        drop0 = tf.keras.layers.Dropout(DROPOUT_VALUE)(output0.output)
        drop1 = tf.keras.layers.Dropout(DROPOUT_VALUE)(output1.output)
        drop2 = tf.keras.layers.Dropout(DROPOUT_VALUE)(conv1)
        drop3 = tf.keras.layers.Dropout(DROPOUT_VALUE)(conv3)

        class0 = tf.keras.layers.Conv2D(4*output_channels,3,strides=1,padding='same',kernel_initializer='he_normal')(drop0)#0
        class1 = tf.keras.layers.Conv2D(4*output_channels,3,strides=1,padding='same',kernel_initializer='he_normal')(drop1)#1
        class2 = tf.keras.layers.Conv2D(4*output_channels,3,strides=1,padding='same',kernel_initializer='he_normal')(drop2)#2
        class3 = tf.keras.layers.Conv2D(4*output_channels,3,strides=1,padding='same',kernel_initializer='he_normal')(drop3)#3

        class0_flatten = tf.keras.layers.Flatten()(class0)
        class1_flatten = tf.keras.layers.Flatten()(class1)
        class2_flatten = tf.keras.layers.Flatten()(class2)
        class3_flatten = tf.keras.layers.Flatten()(class3)

        # concatenate all the classifiers
        conc = K.concatenate([class0_flatten,
                              class1_flatten,
                              class2_flatten,
                              class3_flatten], axis=-1)
        out = tf.keras.layers.Reshape((-1, output_channels))(conc)

        mm = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=out
            )

        return mm

    def __init__(self, data_dir_, box_converter_):
        self.data_dir = data_dir_
        self.box_converter = box_converter_

        # prepare dataset
        self._prepareDataset()

        # prepare model
        self.model = self._buildModel()
        #self.model.summary()

    def train(self, learning_rates, epochs=40):
        drawCallback = ShowPredCallback(epochs, self.check_image, self.check_label, self.box_converter, learning_rates)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[0]),loss=Loss)
        history = self.model.fit(self.dataset_train, epochs=epochs,
                        callbacks=[drawCallback],
                        validation_data=self.dataset_valid)

    def showTestResults(self, is_dataset_test=True):
        if is_dataset_test:
            for test_image, test_label in self.dataset_test.take(1):
                pass
        else:
            for test_image, test_label in self.dataset_train.take(1):
                pass

        for TEST_IMAGE_NUMBER in range(test_image.shape[0]):
            for_test = test_image[TEST_IMAGE_NUMBER]
            for_test = tf.reshape(for_test, [1,image_size,image_size,3])

            predicted = self.model.predict(for_test)

            recognized_data = self.box_converter.get_best_boxes(predicted[0], output_channels,0.5)

            recognized_data = np.array(non_max_suppression(recognized_data, 0.1, 0.65))# nms
            for dt in recognized_data:
                print(dt)
            print("-----------")

            # clone image
            image = for_test[0].numpy().copy()

            # draw
            draw_all_rectanges(image, recognized_data)

            image = cv2.resize(image, [600,600])
            cv2.imshow("check image", image)
            cv2.waitKey(1001)
        cv2.destroyAllWindows()


    def convertToSupportedFormat(self):
        """
            Installed OpenCV doesn't support reshape layers.
            This conversion instead of reshaping, concatenates outputs.

        """
        # convert
        out0 = self.model.layers[-6]
        out1 = self.model.layers[-5]
        out2 = self.model.layers[-4]
        out3 = self.model.layers[-3]

        conc = K.concatenate([out0.output,out1.output, out2.output, out3.output], axis=-1)

        supported_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[conc]
            )

        return supported_model
        
"""
# VGG16 SSD 300
x1 = tf.keras.layers.InputLayer(input_shape=[300, 300, 3])
#x1 = Input(shape=(300, 300, 3))
conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_1')(x1.output)
conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_2')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_1')(pool1)
conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_2')(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3_1)
conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_3')(conv3_2)
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4_1)
conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_3')(conv4_2)
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5_1)
conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_3')(conv5_2)
pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', name='fc6')(pool5)

fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name='fc7')(fc6)

conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name='conv6_1')(fc7)
conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', name='conv6_2')(conv6_1)

conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name='conv7_1')(conv6_2)
conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', name='conv7_2')(conv7_1)

conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name='conv8_1')(conv7_2)
conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', name='conv8_2')(conv8_1)

conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name='conv9_1')(conv8_2)
conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', name='conv9_2')(conv9_1)
#model = tf.keras.models.Model(inputs=x1.input, outputs=conv9_2)


# We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
# Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
# We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
# Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`

class0 = layers.Conv2D(4*15,3,padding='same')(conv4_3)#0
class1 = layers.Conv2D(4*15,3,padding='same')(fc7)#1
class2 = layers.Conv2D(4*15,3,padding='same')(conv6_2)#2
class3 = layers.Conv2D(4*15,3,padding='same')(conv7_2)#3
class4 = layers.Conv2D(4*15,3,padding='same')(conv8_2)#4
class5 = layers.Conv2D(4*15,3,padding='same')(conv9_2)#5

class0_resh = layers.Reshape((-1, 15))(class0)
class1_resh = layers.Reshape((-1, 15))(class1)
class2_resh = layers.Reshape((-1, 15))(class2)
class3_resh = layers.Reshape((-1, 15))(class3)
class4_resh = layers.Reshape((-1, 15))(class4)
class5_resh = layers.Reshape((-1, 15))(class5)

# concatenate all the classifiers
out = layers.concatenate([class0_resh,class1_resh,class2_resh,class3_resh,class4_resh,class5_resh], axis = -2, name='concatenate')

mm = tf.keras.models.Model(
    inputs=x1.input,
    outputs=out
    )
mm.summary()
"""
