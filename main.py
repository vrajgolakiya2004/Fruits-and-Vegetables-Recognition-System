import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import os


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(
        r"D:\Python Practice\Fruits and Vegetables Recognition System\trained_model.h5"
    )
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = r"D:\Python Practice\Fruits and Vegetables Recognition System\Download_image\home_img.jpg"
    st.image(image_path, use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code(
        "fruits - banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango."
    )
    st.code(
        "vegetables - cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant."
    )
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")

            # Save uploaded file temporarily for prediction
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(test_image.read())
                temp_path = tmp_file.name

            result_index = model_prediction(temp_path)

            # Clean up temp file
            os.remove(temp_path)

            # Reading Labels
            with open(
                r"D:\Python Practice\Fruits and Vegetables Recognition System\labels.txt"
            ) as f:
                content = f.readlines()

            label = [i.strip() for i in content]
            st.success(f"Model is Predicting it's a **{label[result_index]}**")
