import streamlit as st
import glob
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

st.title("USHMM Image Processing Pipeline")

# @st.cache(allow_output_mutation=True)
# def load_my_model():
#     MODEL_PATH = "models/01_image_type"
#     model = load_model(MODEL_PATH)
#     model._make_predict_function()
#     model.summary()  # included to make it visible when model is reloaded
#     return model

# model = load_my_model()

MODEL_PATH = "models/01_image_type"
model = load_model(MODEL_PATH)

def test_model(classes, file, threshold):
    heldout_files = glob.glob(heldout_files)
    image_size = (224, 224)
    i=0
    results = {"unknown": []}
    for c in classes:
        results[c] = []
    img = load_img(file, target_size=image_size)

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    scores = predictions[0]

    found = False
    i=0
    for score in scores:
        if score > threshold:
            results[classes[i]].append(img)
            found=True
        i=i+1
    if found == False:
        results["unknown"].append(img)
    print ("Complete")
    for item in results:
        print (item, len(results[item]))
    return results