import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os


def load_dataset(dataset_path):
    category_dict = {}
    for root, dirs, files in os.walk(dataset_path):
        for name in dirs:
            category_dict[name] = []
        ordered_files = files.copy()
        ordered_files.sort()
        for name in ordered_files:
            if name.endswith('.jpg') or name.endswith('.png'):
                category = os.path.basename(root)
                if category_dict.get(category) is not None:
                    im_numpy = cv2.imread(os.path.join(root, name))
                    if im_numpy is not None:
                        category_dict[category].append(im_numpy)

    if len(category_dict) == 0:
        raise FileNotFoundError('No file found')

    category_name_list = list(category_dict.keys())
    category_name_list.sort()

    return category_name_list, category_dict


def prepare_dataset_for_training(category_dict, category_name_list, input_size, nb_valid_per_category=2, seed=33):
    train_raw_dict = {}
    x_valid_list = []
    y_valid_list = []

    np.random.seed(seed) # not used in this version of the algorithm
    # use list to keep order for determinism of the randomness!
    for category in category_name_list:
        image_list = category_dict[category].copy()
        for image in image_list[-nb_valid_per_category:]:
            x_valid_list.append(preprocess_image(image, input_size))
            y_valid_list.append(to_categorical(category_name_list.index(category), len(category_name_list)))
        train_raw_dict[category] = image_list[:-nb_valid_per_category]

    x_valid = np.array(x_valid_list)
    y_valid = np.array(y_valid_list)

    return train_raw_dict, x_valid, y_valid


def preprocess_image(img, size):
    resized = cv2.resize(img, size)
    normalized = resized / 127.5
    normalized -= 1
    return normalized


def preprocess_image_with_da(i, size, da_factor):
    # random resize
    rand_width = np.random.randint(size[0] + 2, int(size[0] * (da_factor ** 3)))
    rand_size = (rand_width, rand_width)
    resized = cv2.resize(i, (rand_size[0], rand_size[1]))

    # random rotate
    theta = np.random.randint(360)
    scale = np.random.randint(int(100 / da_factor), int(100 * da_factor)) / 100
    kernel_rotation = cv2.getRotationMatrix2D(((rand_size[0] - 1) / 2.0, (rand_size[1] - 1) / 2.0), theta, scale)
    rotated = cv2.warpAffine(resized, kernel_rotation, rand_size,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # crop
    x = np.random.randint(0, rotated.shape[0] - size[0])
    y = np.random.randint(0, rotated.shape[1] - size[1])
    cropped = rotated[x:x + size[0], y:y + size[1]]

    # brightness
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    random_br = np.random.randint(int(100 / da_factor), int(100 * da_factor)) / 100
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel
    brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # normalize
    normalized = brightness / 127.5
    normalized -= 1

    # random flip
    x_flip = 1 - np.random.randint(2) * 2
    y_flip = 1 - np.random.randint(2) * 2
    flipped = normalized[::x_flip, ::y_flip, :]
    return flipped


def balanced_da_generator(img_dict, keys_list, batch_size, size, da_factor):
    while True:
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            y = np.random.randint(len(keys_list))
            key = keys_list[y]
            i = np.random.randint(len(img_dict[key]))
            image = img_dict[key][i]
            x_batch.append(preprocess_image_with_da(image, size, da_factor))
            y_batch.append(to_categorical(y, len(keys_list)))
        yield (np.array(x_batch), np.array(y_batch))
