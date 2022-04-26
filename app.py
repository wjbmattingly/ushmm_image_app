import streamlit as st
import glob
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
from operator import itemgetter
import pandas as pd




@st.cache(allow_output_mutation=True)
def load_my_model():
    MODEL_PATH = "models/01_image_type_15"
    model = load_model(MODEL_PATH)

    return model

def test_model(classes, file, threshold, model):
    image_size = (224, 224)
    i=0
    results = []
#     for c in classes:
#         results[c] = 0
    img = load_img(file, target_size=image_size)

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    scores = predictions[0]
#     st.write(scores)
    found = False
    i=0
    for score in scores:
        if score > threshold:
            results.append((classes[i], score))
            found=True
        i=i+1
    if found == False:
        results.append(("unknown", 1))
    print ("Complete")
    results = sorted(results, key=itemgetter(1))
    results.reverse()
    return results
model = load_my_model()
st.title("USHMM Image Processing Pipeline")
image = st.file_uploader("Load your Image")

if st.button("Run"):
    image = Image.open(image)

    col1, col2 = st.columns(2)
    col1.image(image, caption='Your Image')
    image = image.convert("RGB")
    image.save("temp_image.jpg")


    classes = ['document', 'passport', 'photograph']


    results = test_model(classes, "temp_image.jpg", .0, model)
    col2.write(pd.DataFrame(results, columns=["class", "score"]))
