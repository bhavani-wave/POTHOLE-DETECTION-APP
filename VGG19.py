import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("VGG19.h5")

# Define the class labels
classes = ['not pothole', 'pothole']

# Define a function to preprocess the image and make predictions
def predict(image):
    # Load and preprocess the image
    img = load_img(image, target_size=(128, 128))
    x = img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    # Make predictions
    pred = model.predict(x)
    pred_class = np.argmax(pred, axis=1)
    return classes[pred_class[0]]

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Pothole Detection', page_icon=':car:', layout='wide')
    st.title('Pothole Detection')
    # Upload an image
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded image', use_column_width=True)
        # Make predictions and display the result
        pred = predict(uploaded_file)
        st.success(f'The image is classified as: {pred}')

# Run the Streamlit app
if __name__ == '__main__':
    app()
