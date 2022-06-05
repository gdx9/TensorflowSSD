import cv2
import numpy as np
import random
import time
import string

from IPython.display import clear_output

import os
from os import listdir
from os import path
from os.path import isfile, join

class ImageClassData:
    def __init__(self, class_num_, class_name_, width_range_, height_range_, image_folder_path, size):
        self.class_num    = class_num_
        self.class_name   = class_name_
        self.width_range  = width_range_
        self.height_range = height_range_
        self.images       = list()

        if image_folder_path is not None:
            all_image_pathes = [join(image_folder_path, f) for f in listdir(image_folder_path) if isfile(join(image_folder_path, f))]

            # background images
            for image_path in all_image_pathes:
                self.images.append(self._prepareImage(image_path, size))

    def __str__(self):
        return "class {:d}, class name: {}, images number: {:d}, width_range: {}, height_range: {}".format(
            self.class_num,
            self.class_name,
            len(self.images),
            self.width_range,
            self.height_range)

    def _prepareImage(self, path, size=None):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if size is not None:
            image = cv2.resize(image, size)

        # to RGBA
        if image.shape[2] != 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # make transparent pixels black
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y,x,3] != 255:
                    image[y,x] = (0,0,0,0)

        return image

class SyntheticDatasetGenerator:

    def __init__(self):
        """
            prepare all images for generation
        """

        BACKGROUND_IMAGE_SIZE = (640,480)
        CLASS_IMAGE_SIZE = (300,300)

        # background images
        self.backgrounds = ImageClassData(7, "background", (1,10), (10,30), "back_images", BACKGROUND_IMAGE_SIZE)
        self.image_classes =[
            ImageClassData(0,  "analog_clock", (200,260), (200,260),   "image_classes/analog_clock", CLASS_IMAGE_SIZE),
            ImageClassData(1,  "apple",        (180, 200), (180, 200), "image_classes/apple",        CLASS_IMAGE_SIZE),
            ImageClassData(2,  "bottle",       (110,130), (330,370),   "image_classes/bottle",       CLASS_IMAGE_SIZE),
            ImageClassData(3,  "cat",          (330,370), (310,330),   "image_classes/cat",          CLASS_IMAGE_SIZE),
            ImageClassData(4,  "cup",          (110,130), (330,370),   "image_classes/cup",          CLASS_IMAGE_SIZE),
            ImageClassData(5,  "key",          (200,260), (100,160),   "image_classes/key",          CLASS_IMAGE_SIZE),
            ImageClassData(6,  "spider",       (200,260), (200,260),   "image_classes/spider",       CLASS_IMAGE_SIZE)
        ]

        for image_class in self.image_classes:
            print(image_class)

    def _drawModifiedClassImage(self, generated,
                                label_data, image_class_data,
                                GENERATE_THRESHOLD=0.5):
        """
            draw image and add its data to label_data
        """
        show_probability = random.random()

        if show_probability > GENERATE_THRESHOLD:
            # random image
            image_num = np.random.randint(len(image_class_data.images))
            img = image_class_data.images[image_num]

            img_modified = self._modifyClassImage(img,
                                                  image_class_data.width_range,
                                                  image_class_data.height_range)
            start_x,start_y = self._drawImageOnBackground(generated, img_modified,
                                               alpha=random.uniform(0., 0.15))

            data_string = str(image_class_data.class_num) + " " + str(start_x) + " " + str(start_y) + " " + str(img_modified.shape[1]) + " " + str(img_modified.shape[0])
            label_data.append(data_string)

    def _modifyClassImage(self, class_image, width_range, height_range):
        cls_img = self._getRotatedImage(class_image,
                                angle=random.randrange(-5, 5),# +/- 5
                                )
        cls_img = self._warpClockImage(cls_img)

        # crop only clock (without empty area)
        (x_min, y_min), (x_max,y_max) = self._getNonZeroAreaBorders(cls_img)
        cls_img = cls_img[y_min:y_max,x_min:x_max]

        cls_img = self._resizeImage(cls_img, width_range, height_range)

        return cls_img

    def _getRotatedImage(self, image, angle, center=None, scale=1.0):
        """
            rotate image
            Note: angle - counterclockwise
        """
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0,0,0,0))

        return rotated


    def _warpClockImage(self, src):
        """
            warp clock image
        """
        srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]).astype(np.float32)

        x_shift_1 = random.randrange(0, 4)    * src.shape[1]/100
        y_shift_1 = random.randrange(0, 4)    * src.shape[0]/100
        x_shift_2 = random.randrange(60, 101) * src.shape[1]/100
        y_shift_2 = random.randrange(-5, 6)   * src.shape[0]/100
        x_shift_3 = random.randrange(0, 3)    * src.shape[1]/100
        y_shift_3 = random.randrange(70,100)  * src.shape[0]/100

        dstTri = np.array( [[0+x_shift_1, y_shift_1],
                            [x_shift_2, y_shift_2],
                            [x_shift_3, y_shift_3]
                            ] ).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]), borderValue=(0,0,0,0))

        return warp_dst

    def _getNonZeroAreaBorders(self, image):

        i,j = np.nonzero(image[...,3])
        y_min = i[np.argmin(i)]
        x_min = j[np.argmin(j)]
        y_max = i[np.argmax(i)]
        x_max = j[np.argmax(j)]
        return (x_min, y_min), (x_max,y_max)

    def _randomRangeBetween(self, min_val, max_val):
        return min_val + random.randrange(0,max_val-min_val)

    def _resizeImage(self, img, width_range, height_range):
        """
            resize image randomly
        """

        new_width  = self._randomRangeBetween(width_range[0],  width_range[1])
        new_height = self._randomRangeBetween(height_range[0], height_range[1])

        img = cv2.resize(img, (new_width, new_height))

        return img

    def _drawImageOnBackground(self, background, draw_image, alpha=0.8):
        """
            draw clock on background, in random position
            and
            prepare a mask for that clock
        """
        start_y = random.randrange(0, background.shape[0]-draw_image.shape[0]-1)
        start_x = random.randrange(0, background.shape[1]-draw_image.shape[1]-1)

        beta = 1. - alpha

        for y in range(draw_image.shape[0]):
            for x in range(draw_image.shape[1]):
                if draw_image[y,x,3] != 0:
                    # mix colors
                    background[y+start_y,x+start_x] = background[y+start_y,
                                                                 x+start_x] * alpha + draw_image[y,x] * beta

        return start_x,start_y

    def _apply_salt_pepper_noise(self,image, probability):
        """
            apply salf and pepper noise on image
        """
        probs = np.random.random(image.shape[:2])
        # rgba
        black = np.array([0, 0, 0, 255], dtype=np.uint8)
        white = np.array([255, 255, 255, 255], dtype=np.uint8)

        image[probs < (probability / 2)] = black
        image[probs > 1 - (probability / 2)] = white

    def _writeTxtFileData(self, filename, lines):
        with open(filename, 'w') as f:
            n = 0
            for line in lines:
                f.write(line)
                n += 1
                if n < len(lines):
                    f.write("\n")


    def generateDataset(self, save_dir, generate_number, save=False):
        """
            average time: 1000 images in 6 minutes
        """
        print("to generate:", generate_number)

        start_time = time.time()

        for image_number in range(generate_number):
            # get random background
            background_num = np.random.randint(len(self.backgrounds.images))
            generated = self.backgrounds.images[background_num].copy()

            label_data = list()

            for image_class in self.image_classes:
                self._drawModifiedClassImage(generated, label_data, image_class)

            # add noise to image
            self._apply_salt_pepper_noise(generated, probability=random.uniform(0.001, 0.01))

            #print(label_data)

            if save == True:
                # saving data
                path_img = join(save_dir, str(image_number) + '.png')
                path_txt = join(save_dir, str(image_number) + '.txt')
                cv2.imwrite(path_img, generated)
                self._writeTxtFileData(path_txt, label_data)
            else:
                cv2.imshow("generated", generated)
                cv2.waitKey(1333)

            # logging
            clear_output(wait=True)
            percentage = (image_number+1) / generate_number * 100
            print("processed {perc:.1f}% out of {gen_num} images".format(perc=percentage, gen_num=generate_number))

        end_time = time.time()
        seconds = end_time - start_time
        print('finished generating {im_num:d} pictures in {sec:.2f} seconds'.format(
            im_num=image_number+1, sec=seconds))

        cv2.destroyAllWindows()
