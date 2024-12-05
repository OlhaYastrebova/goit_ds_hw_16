import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
from os.path import exists
import gdown
import tensorflow as tf

# Заголовок сторінки
st.title("Класифікація зображень за допомогою CNN та VGG16")

# Вибір моделі
st.sidebar.header("Виберіть модель")
selected_model_name = st.sidebar.selectbox(
    "Моделі",
    ["CNN", "VGG16"]
)

# Завантаження моделей
@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model.h5")

@st.cache_resource
def load_vgg16_model():
    return load_model("vgg16_model.h5")

def get_vgg16_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = "models/vgg16.keras"
    if not exists(model_path):
        url = "https://drive.google.com/file/d/19fUTQSt9hYdln8qvGl4k1uIWPuWzYkYz/view?usp=drive_link"
        output = model_path
        gdown.download(url, output, quiet=False, fuzzy=True)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        model = tf.keras.models.load_model(model_path)
        return model

# Завантаження відповідної моделі
if selected_model_name == "CNN":
    model = load_cnn_model()
    target_size = (28, 28)
    color_mode = "grayscale"
elif selected_model_name == "VGG16":
    model = get_vgg16_model()
    target_size = (32, 32)
    color_mode = "rgb"

# Завантаження зображення
uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Відображення завантаженого зображення
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_container_width=True)

    # Попередня обробка зображення
    img = load_img(uploaded_file, target_size=target_size, color_mode=color_mode)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Передбачення
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Вивід результатів
    st.subheader(f"Результати класифікації ({selected_model_name})")
    for idx, prob in enumerate(predictions[0]):
        st.write(f"Клас {idx}: {prob:.2f}")
    st.write(f"Передбачений клас: {predicted_class[0]}")

# Інформація про графіки
st.sidebar.header("Інформація про модель")
show_metrics = st.sidebar.checkbox("Показати графіки точності та функції втрат")

if show_metrics:
    st.subheader("Графіки функції втрат та точності")
    
    # Заглушка для графіків. Замініть на реальні дані.
    # Якщо у вас є файл з історією тренування (наприклад, history.pickle):
    # 1. Завантажте його і витягніть значення.
    # 2. Побудуйте графіки для втрат та точності.
    epochs = np.arange(1, 11)
    loss = [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.2, 0.18, 0.15]
    accuracy = [0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(epochs, loss, label="Втрата", color="red")
    ax[0].set_title("Функція втрат")
    ax[0].set_xlabel("Епохи")
    ax[0].set_ylabel("Втрати")
    ax[0].legend()

    ax[1].plot(epochs, accuracy, label="Точність", color="blue")
    ax[1].set_title("Точність")
    ax[1].set_xlabel("Епохи")
    ax[1].set_ylabel("Точність")
    ax[1].legend()

    st.pyplot(fig)


