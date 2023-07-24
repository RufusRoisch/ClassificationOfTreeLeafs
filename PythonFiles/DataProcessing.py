import time
import numpy as np
from PIL import Image


# gets a single leaf picture from resources
# is used by get_data()
def get_picture(tree_class_num: int, leaf_num: int, data_id: int):
    # constructs path to picture from tree_num and leaf_num
    source_dir = "Resources/LeafPicturesDownScaled"
    addon = ""
    if data_id == 2:
        source_dir = "Resources/LeafPicsDownScaled_hsv"
        addon = "_hsv"
    elif data_id == 3:
        source_dir = "Resources/LeafPicsDownScaled_bw"
        addon = "_bw"

    img_name = "l" + str(tree_class_num) + "nr" + f"{leaf_num:03d}" + addon + ".tif"
    img_path = source_dir + "/" + img_name

    # loads the image from path as a numpy array and returns it
    img = np.array(Image.open(img_path))

    return img


def get_data_bw():
    return get_data(3)


def get_data_hsv():
    return get_data(2)


def get_data_normal():
    return get_data(1)


# returns numpy arrays training_data, training_labels, validation_data, validation_labels
# data_id: 1 = normal_data, 2 = hsv_data, 3 = bw_data
def get_data(data_id: int):
    # the time at which get_data() got called first
    start_time = time.time()

    # initializing data lists
    training_img_list = []
    training_label_list = []
    validation_img_list = []
    validation_label_list = []

    # iterates over all leaf pictures of every tree class
    for tree_class in range(1, 16, 1):
        for leaf_num in range(1, 76, 1):
            # gets next picture
            new_pic = get_picture(tree_class, leaf_num, data_id)
            # puts first 60 leafs and their label of each class into training data
            if leaf_num <= 60:
                training_img_list.append(new_pic)
                training_label_list.append(tree_class)
            # puts the last 15 leafs and their label of each class into test data
            else:
                validation_img_list.append(new_pic)
                validation_label_list.append(tree_class)
    # returns collected and divided data
    return (
        np.array(training_img_list),
        np.array(training_label_list),
        np.array(validation_img_list),
        np.array(validation_label_list),
    )
