import cv2
import numpy as np
from descriptores import Pixel


class LBPUDescriptor:

    def __init__(self, window_size: int = 3):
        """
        Inicialización de los parámetros de entrada.
        Realmente el código está solo preparado para la implementación básica con window_size = 3
        """
        self._window_size = window_size
        self._border = window_size // 2

    def _uniform_neighborhood_comparation(self, window_list_values: list, pixel_value: int) -> int:
        """
        Para una ventana en formato lista de valores y el valor del píxel central, se calcula el valor decimal
        correspondiente al valor binario resultante de LBP Uniforme.
        x1y1 x1y2 x1y3
        x2y1 x2y2 x2y3 -> [x1y1, x1y2, x1y3, x2y1 ,x2y2 ,x2y3 ,x3y1 ,x3y2 ,x3y3]
        x3y1 x3y2 x3y3

        ...

        Attributes
        ---
            window_list_values : list
                Valores de la ventana a calcular.

            pixel_value: int
                Valor del píxel central.

        ...

        Returns
        ---
            Devuelve el valor LBP Uniforme asociado al pixel central
        """
        values = [1 if pixel > pixel_value else 0 for pixel in window_list_values]
        values_without_center = [values[i] for i in [2, 5, 8, 7, 6, 3, 0, 1]]
        values_changing = [value for index, value in enumerate(values_without_center) if value != values_without_center[index - 1]]
        if len(values_changing) <= 2:
            label = int("".join(str(a) for a in values_without_center), 2)
        else:
            label = -1

        return label

    def _calculate_windows_list_format(self, img: np.ndarray, pixel: Pixel):
        """
        Calcula para una imágen dada y un pixel determinado su ventana de vecinos.

        ...

        Attributes
        ---
            img : np.ndarray
                imagen original.
            pixel : Pixel
                pixel central al que calcular el valor.

        Returns
        ---
            Devuelve la ventana window_size*window_size correspondiente al pixel central.

        """

        combinaciones_x_y = [(x, y) for y in range(pixel.y - self._border, pixel.y + self._border + 1)
                             for x in range(pixel.x - self._border, pixel.x + self._border + 1)]
        return [img[y, x] for x, y in combinaciones_x_y]


    def compute_lbpu_image(self, img : np.ndarray):
        """
        Calcula la imagen lbp uniforme correspondiente a la imagen original.

        Attributes
        ---
            img : np.ndarray
                imagen original.

        Returns
        ---
            Imagen LBP Uniforme asociada.

        """

        rows, columns = img.shape
        indexs = [Pixel(x, y) for y in range(self._border, rows - self._border) for x in
                  range(self._border, columns - self._border)]

        pixel_value_with_windows = [(img[p.y, p.x], self._calculate_windows_list_format(img, p)) for p in indexs]

        lbpu_image = np.float32(
            [self._uniform_neighborhood_comparation(window_list_values=w[1], pixel_value=w[0]) for w in
             pixel_value_with_windows])

        return lbpu_image


    def compute(self, img : np.ndarray):
        """
        Calcula el histograma LBP Uniforme y por tanto el descriptor de una imagen.

        Attributes
        ---
            img : np.ndarray
                imagen original.

        Returns
        ---
            Descriptor LBP Uniforme de la imagen asociada.
        """
        lbpu_img = self.compute_lbpu_image(img)
        lbpu_descriptor = np.float32(
            [len(np.where(lbpu_img == value)[0]) for value in POSSIBLE_UNIFORM_VALUES]
        )
        return lbpu_descriptor


POSSIBLE_UNIFORM_VALUES = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112,
                           120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
                           225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
