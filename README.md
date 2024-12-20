# **📊 Extracción de Rasgos**

- **👨‍🎓 Alumno:** Álvaro Santana Sánchez  
- **📅 Fecha:** Diciembre 2024  
- **💻 IDE:** PyCharm
- **🐱 GIT:** https://github.com/sanntana21/image_feature_extraction

## **📂 Estructura del Dataset**  
Las imágenes deben organizarse con la siguiente estructura desde el directorio raíz:  

- **PATH_POSITIVE_TRAIN** = "kmnist\\train\\3\\"
- **PATH_NEGATIVE_TRAIN** = "kmnist\\train\\7\\"
- **PATH_POSITIVE_TEST** = "kmnist\\test\\3\\"
- **PATH_NEGATIVE_TEST** = "kmnist\\test\\7\\"
- **IMAGE_EXTENSION** = ".png"

## **📜 Archivos de la Práctica**  

- 📒 `main.ipynb`: Notebook con los valores definidos en el PDF.  
- 🛠️ `descriptores.py`: Implementación de descriptores (HOG, LBP + clase PIXEL 2D).  
- 🔍 `LBPUDescriptor.py`: Descriptor LBP uniforme.  
- 🧰 `utils.py`: Funcionalidades auxiliares.  
- 📦 `requirements.txt`: Instalación de dependencias.  
- 📥 `descargar_datasets.py`: Descarga los archivos `.npz` de todas las clases (train y test).  
- 📤 `split_datasets.py`: Divide los datasets en carpetas `.png` según clase.  


## **⚙️ Requisitos**  
Instala las dependencias necesarias ejecutando:  
```bash
pip install -r requirements.txt