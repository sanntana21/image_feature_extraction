import os
import numpy as np
from PIL import Image
"""

SCRIPT UTILIZADO PARA DIVIDIR EL DATASET EN CARPETAS

TRAIN:
    0
    1
    ...
TEST:
    0
    1
    ...

"""


def save_images_by_class(image_file, class_file, output_folder, dataset_name):
    images = np.load(image_file)['arr_0']
    labels = np.load(class_file)['arr_0']


    base_folder = os.path.join(output_folder, dataset_name)
    os.makedirs(base_folder, exist_ok=True)

    for i, (img, label) in enumerate(zip(images, labels)):
        class_folder = os.path.join(base_folder, str(label))
        os.makedirs(class_folder, exist_ok=True)
        img = Image.fromarray(img, mode='L')
        img.save(os.path.join(class_folder, f'image_{i}.png'))

    print(f"{dataset_name} images saved to {base_folder}.")


train_images_file = 'kmnist\\kmnist-train-imgs.npz'
train_classes_file = 'kmnist\\kmnist-train-labels.npz'
test_images_file = 'kmnist\\kmnist-test-imgs.npz'
test_classes_file = 'kmnist\\kmnist-test-labels.npz'

output_folder = 'kmnist'

# Process train and test datasets
save_images_by_class(train_images_file, train_classes_file, output_folder, 'train')
save_images_by_class(test_images_file, test_classes_file, output_folder, 'test')
