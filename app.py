import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Load your trained model
model = tf.keras.models.load_model("pothole_detection12_model (1).h5")

# Define the labels
labels = ['not_pothole', 'pothole']

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image
    image = image.resize((128, 128))
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # Convert the RGB image to BGR
    img_array = img_array[:, :, ::-1]
    # Convert the pixel values to the range [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the Streamlit app
def main():
    # Set the title and page icon
    st.set_page_config(page_title='Pothole Detector', page_icon=':car:')

    # Add a title to the app
    st.title('Pothole Detector')

    # Add a file uploader to the app
    uploaded_file = st.file_uploader('Upload an image of the road', type=['jpg', 'jpeg', 'png'])

    # If a file is uploaded
    if uploaded_file is not None:
        # Open the uploaded image with PIL
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Use the model to predict the class probabilities
        predictions = model.predict(preprocessed_image)[0]

        # Get the predicted label
        predicted_label = labels[np.argmax(predictions)]

        # Display the uploaded image and the predicted label
        st.image(image, caption=f'Uploaded Image ({predicted_label})', use_column_width=True)
        st.write(predicted_label)
        


# Run the app
if __name__ == '__main__':
    main()





