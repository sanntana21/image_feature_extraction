from typing import Optional
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

import imutils
from descriptores import Pixel, HogDescriptor

PATH_POSITIVE_TRAIN = "kmnist\\train\\3\\"
PATH_NEGATIVE_TRAIN = "kmnist\\train\\7\\"
PATH_POSITIVE_TEST = "kmnist\\test\\3\\"
PATH_NEGATIVE_TEST = "kmnist\\test\\7\\"
IMAGE_EXTENSION = ".png"


def load_and_transform(path):
    """
    Carga las imágenes con las transformaciones definidas
    """
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def cargar_imagenes(test : bool = False):
    PATH_POSITIVE = PATH_POSITIVE_TEST if test else PATH_POSITIVE_TRAIN
    PATH_NEGATIVE = PATH_NEGATIVE_TEST if test else PATH_NEGATIVE_TRAIN
    output = "test" if test else "entrenamiento"
    training_data = []
    classes = []

    # Casos positivos
    counter_positive_samples = 0
    for filename in tqdm(os.listdir(PATH_POSITIVE)):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_POSITIVE + filename
            gray_img = load_and_transform(filename)
            training_data.append(gray_img)
            classes.append(1)
            counter_positive_samples += 1

    print(f"Leidas  {str(counter_positive_samples)} imágenes de {output} -> positivas")

    # Casos negativos
    counter_negative_samples = 0
    for filename in tqdm(os.listdir(PATH_NEGATIVE)):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_NEGATIVE + filename
            gray_img = load_and_transform(filename)
            training_data.append(gray_img)
            classes.append(0)
            counter_negative_samples += 1

    print(f"Leidas  {str(counter_positive_samples)} imágenes de {output} -> negativas")

    return np.array(training_data), np.array(classes)


def ejemplo_clasificador_imagenes():
    """

    Prueba de entrenamiento de un clasificador
    """
    # Obtenemos los datos para el entrenamiento del clasificador
    training_data, classes = load_training_data()
    # Entrenamos el clasificador
    clasificador = train(training_data, classes)
    # Leemos imagen a clasificar
    image = cv2.imread(EXAMPLE_POSITIVE)
    if image is None:
        print("Cannot load image ")

    # Clasificamos
    prediccion = test(image, clasificador)
    print("Predicción: " + str(prediccion))


def compute_image_descriptors(images: np.array, descriptor_model: Optional = None,
                              ):
    """
    Lee las imágenes pasasdas como argumento y calcula sus descriptores para el entrenamiento.

    returns:
    np.array: numpy array con los descriptores de las imágenes leídas
    """
    if not descriptor_model:
        descriptor_model = HogDescriptor()

    data = []
    print("Computando descriptores")
    for image in tqdm(images):
        data.append(descriptor_model.compute(image))

    return np.array(data)


def load_training_data(descriptor_model: Optional = None):
    """
    Lee las imágenes de entrenamiento (positivas y negativas) y calcula sus
    descriptores para el entrenamiento.

    returns:
    np.array: numpy array con los descriptores de las imágenes leídas
    np.array: numpy array con las etiquetas de las imágenes leídas
    """
    if not descriptor_model:
        descriptor_model = HogDescriptor()

    training_data = []
    classes = []
    # Casos positivos
    counter_positive_samples = 0
    for filename in tqdm(os.listdir(PATH_POSITIVE_TRAIN)):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_POSITIVE_TRAIN + filename
            gray_img = load_and_transform(filename)
            descriptor = descriptor_model.compute(gray_img)
            training_data.append(descriptor)
            classes.append(1)
            counter_positive_samples += 1

    print("Leidas " + str(counter_positive_samples) + " imágenes de entrenamiento -> positivas")

    # Casos negativos
    counter_negative_samples = 0
    for filename in tqdm(os.listdir(PATH_NEGATIVE_TRAIN)):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_NEGATIVE_TRAIN + filename
            gray_img = load_and_transform(filename)
            descriptor = descriptor_model.compute(gray_img)
            training_data.append(descriptor)
            classes.append(0)
            counter_negative_samples += 1

    print("Leidas " + str(counter_negative_samples) + " imágenes de entrenamiento -> negativas")

    return np.array(training_data), np.array(classes)


def basic_svm_model():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    return svm

def train(training_data, classes, svm):
    """
        Entrena el clasificador

        Parameters:
        training_data (np.array): datos de entrenamiento
        classes (np.array): clases asociadas a los datos de entrenamiento

        Returns:
        cv2.SVM: un clasificador SVM
    """
    if not svm:
        svm = basic_svm_model()
    svm.train(training_data, cv2.ml.ROW_SAMPLE, classes)

    return svm


def test(image, clasificador):
    """
    Clasifica la imagen pasada por parámetro

    Parameters:
    image (np.array): imagen a clasificar
    clasificador (cv2.SVM): clasificador

    Returns:
        int: clase a la que pertenece la imagen (1|0)
    """
    # HOG de la imagen a testear
    hog = cv2.HOGDescriptor()
    descriptor = hog.compute(image)
    # Clasificación
    # Devuelve una tupla donde el segundo elemento es un array
    # que contiene las predicciones (en nuestro caso solo una)
    # ej: (0.0, array([[1.]], dtype=float32))
    return int(clasificador.predict(descriptor.reshape(1, -1))[1][0][0])


def test_descriptor(descriptor, model):
    # print(descriptor.reshape(1, -1).shape)
    return int(model.predict(descriptor.reshape(1, -1))[1][0][0])


def print_descriptor_histogram(descriptor, title='Histograma del Descriptor HOG'):
    # Dibujar el histograma del descriptor HOG
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(descriptor)), descriptor, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Índice del Bin')
    plt.ylabel('Valor del Bin')
    plt.show()


#DEPRECATED
def manual_cv(X,
              y,
              validation_function: callable,
              k: int = 5,
              test_size: float = 0.25,
              model=None,
              *,
              random_state=0):
    rs = ShuffleSplit(n_splits=k, test_size=test_size, random_state=random_state)
    accuracy_scores = []
    precision = []
    recall = []
    f1 = []
    conf_matrix = []
    better_model = None
    for i, (train_index, validation_index) in enumerate(rs.split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        clf = train(X_train, y_train, model)
        X_validation = X[validation_index]
        y_validation = y[validation_index]
        predictions = [validation_function(x, clf) for x in X_validation]
        accuracy = accuracy_score(y_validation, predictions)
        precision.append(precision_score(y_validation, predictions))
        recall.append(recall_score(y_validation, predictions))
        f1.append(f1_score(y_validation, predictions))
        accuracy_scores.append(accuracy)
        if len(accuracy_scores) > 1:
            if accuracy > max(accuracy_scores):
                better_model = clf
        else:
            better_model = clf

    output = {
        "accuracy": accuracy_scores,
        "precion": precision,
        "recall": recall,
        "f1": f1,
        "cof_matrix": conf_matrix
    }
    return better_model, output


def transform_array_to_img(array):
    #Marco de pixeles a cero
    n = len(array)
    lado = int(np.ceil(np.sqrt(n)))
    array_cuadrado = np.zeros((lado, lado), dtype=array.dtype)
    array_cuadrado.flat[:n] = array

    # Imagen con márco
    array_con_marco = np.pad(array_cuadrado, pad_width=1, mode='constant', constant_values=0)

    return array_con_marco


def probabilistic_test_descriptor(descriptor, model):
    # print(descriptor.reshape(1, -1).shape)
    return model.predict(descriptor.reshape(1, -1), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][0][0]


def extract_windows(image, window_size=(28, 28), step_size=28):
    windows = []
    img_h, img_w = image.shape[:2]
    win_h, win_w = window_size

    for y in range(0, img_h - win_h + 1, step_size):
        for x in range(0, img_w - win_w + 1, step_size):
            aux_window = image[y:y + win_h, x:x + win_w]
            og_frame = (Pixel(x, y), Pixel(x + win_w, y), Pixel(x + win_w, y + win_h), Pixel(x, y + win_h))
            windows.append((aux_window, og_frame))
    return windows


def extract_better_result(og_img, model, descriptor_model, step_size=28, window_size=(28, 28)):
    best_margin_descriptor_result = 0.0
    best_window = None
    og_img_frame = None
    for window_margin_tuple in extract_windows(og_img, window_size=window_size, step_size=step_size):
        window_img = window_margin_tuple[0]
        new_descriptor = descriptor_model.compute(window_img)
        distance_descriptor_margin = probabilistic_test_descriptor(new_descriptor, model)
        if distance_descriptor_margin > best_margin_descriptor_result or best_window is None:
            best_margin_descriptor_result = distance_descriptor_margin
            best_window = window_img
            og_img_frame = window_margin_tuple[1]

    return best_window, best_margin_descriptor_result, og_img_frame


def print_with_margin(image: np.ndarray, frame: tuple[Pixel]):
    """
    Print de una imágen pero con un marco de ceros
    """
    plt.figure()
    x1 = frame[0].x
    y1 = frame[0].y
    x2 = frame[2].x
    y2 = frame[2].y
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      edgecolor='red', facecolor='none', linewidth=2))  # Dibujar el cuadrado
    plt.axis('off')
    plt.imshow(image, cmap='gray')


def pyramid(image, scale=1.5, minSize=(28, 28)):
    """
    Piramide de imágenes en diferentes escalas
    """
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def print_descriptor_histogram2(descriptor, title='Histograma del Descriptor HOG',xlabel: str = 'Intensidad'):
    """
    Calcula un histograma pero limitando el valor 0 para el caso de LBP
    """

    # Dibujar el histograma del descriptor HOG
    plt.figure(figsize=(10, 6))
    max_value = max(descriptor[1:])
    plt.ylim(0, max_value)
    plt.bar(np.arange(len(descriptor)), descriptor, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Cantidad de valores')
    plt.show()



def extract_better_result(og_img, model, descriptor_model, step_size=28, window_size=(28, 28)):
    """
    Extrae la ventana con mejor evaluación
    """
    best_margin_descriptor_result = 0.0
    best_window = None
    og_img_frame = None
    for window_margin_tuple in extract_windows(og_img, window_size=window_size, step_size=step_size):
        window_img = window_margin_tuple[0]
        new_descriptor = descriptor_model.compute(window_img)
        distance_descriptor_margin = model.predict_proba(new_descriptor.reshape(1,-1))[0][0]
        if distance_descriptor_margin > best_margin_descriptor_result or best_window is None:
            best_margin_descriptor_result = distance_descriptor_margin
            best_window = window_img
            og_img_frame = window_margin_tuple[1]

    return best_window, best_margin_descriptor_result, og_img_frame
