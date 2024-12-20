import cv2
import numpy as np

class Pixel:
    """
    Clase básica de pixel 2D
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class HogDescriptor:
    def __init__(self):
        self._winSize = (28, 28)
        self._blockSize = (8, 8)
        self._blockStride = (2, 2)
        self._cellSize = (4, 4)
        self._nbins = 9

        self.HOG_descriptor = cv2.HOGDescriptor(self._winSize, self._blockSize, self._blockStride, self._cellSize,
                                                self._nbins)

    def compute(self, img: np.array):
        return self.HOG_descriptor.compute(img)


class LBPDescriptor:

    def __init__(self, window_size: int = 3):
        """
        Inicialización de los parámetros de entrada.
        Realmente el código está solo preparado para la implementación básica con window_size = 3
        """
        self._window_size = window_size
        self._border = window_size // 2

    def _binary_neighborhood_comparation(self, window_list_values: list, pixel_value: int):
        """
        Para una ventana en formato lista de valores y el valor del píxel central, se calcula el valor decimal
        correspondiente al valor binario resultante de LBP.
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
            Devuelve el valor LBP asociado al pixel central
        """
        values = [1 if pixel > pixel_value else 0 for pixel in window_list_values]
        values_without_center = [values[i] for i in [2, 5, 8, 7, 6, 3, 0, 1]]
        binary_value = int("".join(str(a) for a in values_without_center), 2)
        return binary_value

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

        combinaciones_x_y = [(x, y)
                             for y in range(pixel.y - self._border, pixel.y + self._border + 1)
                             for x in range(pixel.x - self._border, pixel.x + self._border + 1)
                             ]
        return [img[y, x] for x, y in combinaciones_x_y]

    def compute_lbp_image(self, img: np.ndarray):
        """
        Calcula la imagen lbp correspondiente a la imagen original.

        Attributes
        ---
            img : np.ndarray
                imagen original.

        Returns
        ---
            Imagen LBP asociada.

        """
        rows, columns = img.shape
        indexs = [Pixel(x, y) for y in range(self._border, rows - self._border) for x in
                  range(self._border, columns - self._border)]

        pixel_value_with_windows = [(img[p.y, p.x], self._calculate_windows_list_format(img, p)) for p in indexs]

        lbp_image = np.float32(
            [self._binary_neighborhood_comparation(window_list_values=w[1], pixel_value=w[0]) for w in
             pixel_value_with_windows])

        return lbp_image

    def compute(self, img: np.ndarray):
        """
        Calcula el histograma LBP y por tanto el descriptor de una imagen.

        Attributes
        ---
            img : np.ndarray
                imagen original.

        Returns
        ---
            Descriptor LBP de la imagen asociada.
        """
        lbp_img = self.compute_lbp_image(img)
        lbp_descriptor = np.float32(
            [len(np.where(lbp_img == value)[0]) for value in range(0, 256)]
        )
        return lbp_descriptor
