import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

class RecognitionData:
    def __init__(self, class_num_, probability_, start_point_, end_point_):
        self.class_num   = class_num_
        self.probability = probability_
        self.start_point = start_point_
        self.end_point   = end_point_

    def __str__(self):
        return "class {:d} with probability {:.4f}, {}, {}".format(self.class_num, self.probability,
                                                         self.start_point,
                                                         self.end_point)

    #def __eq__(self, other):
    #    if self.class_num == other.class_num and self.start_point == other.start_point and self.end_point == other.end_point:
    #            return True
    #    else:
    #        return False
    #def __gt__(self, other):
    #    return False

COLORS = [
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (0, 191, 255),
    (0, 255, 0),# 4
    (240,255,240),#5
    (255, 255, 0),
    (128, 1, 0),
    (255, 0, 0),
    (0, 0, 0)
]

class_names = [
    'clock',# 0
    'apple',
    'bottle',
    'cat',
    'cup',
    'key',# 5
    'spider',
    'background',# 7
]

def draw_all_rectanges(image, recognized_data_list):
    for rec_data in recognized_data_list:
        start_point = rec_data.start_point
        end_point   = rec_data.end_point
        #print(rec_data.class_num)

        cv2.rectangle(image, start_point, end_point, COLORS[rec_data.class_num], 1)
        cv2.rectangle(image, start_point, (end_point[0], start_point[1]+16), COLORS[rec_data.class_num], 1)

        show_data = "{} {:.1f}%".format(class_names[rec_data.class_num], rec_data.probability*100)
        cv2.putText(image, show_data, (start_point[0]+2,start_point[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)

def IoU(box1, box2):
    """
        calculate IoU between 2 boxes
    """
    box1 = box1.astype(np.float32)
    box2 = box2.astype(np.float32)

    xmin = np.maximum(box1[:,0],box2[:,0])
    xmax = np.minimum(box1[:,2],box2[:,2])
    ymin = np.maximum(box1[:,1],box2[:,1])
    ymax = np.minimum(box1[:,3],box2[:,3])

    intersection = np.abs(np.maximum(xmax-xmin,0) * np.maximum(ymax-ymin,0))
    boxArea1 = np.abs((box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]))
    boxArea2 = np.abs((box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]))
    unionArea = boxArea1 + boxArea2 - intersection

    assert (unionArea > 0).all()

    iou = intersection / unionArea

    return iou

def bestIoU(search_box, boxes_coords, total_boxes_num, threshold=0.7):
    all_found_ious = IoU(np.matlib.repmat(search_box, total_boxes_num, 1), boxes_coords)

    positions = np.argwhere(all_found_ious > threshold)
    # np.argmax(IoU(np.matlib.repmat(searchBox,BOXES , 1) , boxes))

    if len(positions) < 2:
        largest_ious_args = np.argsort(all_found_ious)[::-1]
        positions = largest_ious_args[:2]

    return positions

def non_max_suppression(recognized_data_list, iou_threshold, threshold):
    max_prob_boxes = [dt for dt in recognized_data_list if dt.probability > threshold]
    max_prob_boxes = sorted(max_prob_boxes, key=lambda x: x.probability, reverse=True)

    boxes_after_nms = []
    while max_prob_boxes:
        this_box = max_prob_boxes.pop(0)

        max_prob_boxes = [
            box
            for box in max_prob_boxes
            if box.class_num != this_box.class_num
            or IoU(np.array([[this_box.start_point[0], this_box.start_point[1], this_box.end_point[0], this_box.end_point[1]]]),
                  np.array([[box.start_point[0], box.start_point[1], box.end_point[0], box.end_point[1]]])
                  ) < iou_threshold
        ]
        boxes_after_nms.append(this_box)

    return boxes_after_nms

def smoothL1(x,y,label):
    diff = K.abs(x-y)
    result = K.switch(diff < 1, 0.5 * diff**2, diff - 0.5)

    return K.mean(result)

def confidenceLoss(y,label):
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
    return K.mean(unweighted_loss)

def Loss(gt, y):
    loss = 0

    # localisation loss
    loss += smoothL1(y[:,:,-4:],gt[:,:,-4:],gt[:,:,0:1])

    # confidence loss
    loss += confidenceLoss(y[:,:,:-4],tf.cast(gt[:,:,0],tf.int32))
    return loss
