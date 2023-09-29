from PIL import Image
import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from keras.applications.densenet import preprocess_input

def extract_features(filename):
    # load the model
    model = DenseNet201()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
        return None



# remove start/end sequence tokens from a summary
def cleanup_summary(summary):
    # remove start of sequence token
    index = summary.find('startseq ')
    if index > -1:
        summary = summary[len('startseq '):]
    # remove end of sequence token
    index = summary.find(' endseq')
    if index > -1:
        summary = summary[:index]
    return summary
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for _ in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += '' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':

            break
    return in_text

# load the tokenizer
#tokenizer = load_model(open('tokenizer.pkl', 'rb'))
tokenizer = Tokenizer()
model = load_model('model.h5')

# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model.h5')
# load and prepare the photograph




# Functions
def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success, frame = vidcap.read()
    count = 0
    frame_skip = 1
    while success:
        success, frame = vidcap.read()  # get next frame from video
        cv2.imwrite(r"img/frame%d.jpg" % count, frame)
        if count % frame_skip == 0:  # only analyze every n=300 frames
            # print('frame: {}'.format(count))
            pil_img = Image.fromarray(frame)  # convert opencv frame (with type()==numpy) into PIL Image
            # st.image(pil_img)
        if count > 20:
            break
        count += 1
    preprocessing()


def preprocessing():
    x = tf.io.read_file('img/frame2.jpg')
    x = tf.io.decode_image(x, channels=3)
    x = tf.image.resize(x, [224, 224])
    x = tf.expand_dims(x, axis=0)
    x = tf.keras.applications.densenet.preprocess_input(x)
    return x


def predict(x):
    P = tf.keras.applications.densenet.decode_predictions(model.predict(x), top=1)
    return P


def main():
    st.image('img/pexels-google-deepmind-18069423.jpg', width=700)
    st.title("Image Caption Generator")

    # selected = st.text_input("Search for an Object here....",)
    file = st.file_uploader("Upload A Video ", type=(['mp4', ]))
    if file is not None:  # run only when user uploads video
        vid = file.name
        with open(vid, mode='wb') as f:
            f.write(file.read())  # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
                    unsafe_allow_html=True)  # display file name

        vidcap = cv2.VideoCapture(vid)  # load video from disk
        cur_frame = 0
        success = True



    if st.button("Get captions"):
        output1 = splitting(vid)
        output2 = preprocessing()
        output3= extract_features(output2)
        items = generate_caption(model, tokenizer, output3, max_length)
        description = cleanup_summary(items)
        print(description)
        st.success('The Output is {}'.format(output3))
        st.success("")


footer = """<style>
a:link , a:visited{
color: blue;
background-color: red;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: black;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: blue;
color: black;
text-align: center;
z-index:1;
}
</style>
<div class="footer">
<p>Developed by TAPIWA CHAREKWA</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
