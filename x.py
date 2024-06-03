import cv2
import tensorflow
import os
from PIL import Image
import numpy as np
import streamlit as st

def prediction(model,img):
    dict={
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"neutral",
    5:"sad",
    6:"surprise"
    }
    test_img=cv2.resize(img,(224,224))
    test_input=test_img.reshape(1,224,224,3)
    pr=model.predict(test_input)[0]
    j=0
    p=max(pr)
    for i in pr:
        if i == p:
            return dict[j]
        j=j+1

def main():
    st.title('Emotion Detection')
    col3,col4,col5,col6,col7,col8=st.columns(6)
    with col3:
            original_title = '<h2 style="font-family:Courier;font-size: 17px;">Gyanbardhan</h2>'
            st.markdown(original_title, unsafe_allow_html=True)
    with col4:
        filename = "gyan.jpeg"
        #img = cv2.imread(filename)
        img=Image.open(filename)
        img=img.resize((25,25))
        #img=cv2.resize(img,(50,50))
        #image=Image.open(img)
        st.image(img)
    #st.markdown(
    #"""
    #<link rel="stylesheet" type="text/css" href="Background.jpg">
    #""",
    #unsafe_allow_html=True)
    
    img=st.file_uploader("Upload an Image......",type=["jpg",".webp","jpeg","png"])
    
    
    if img is not None:
        image=Image.open(img)
        img = np.asarray(image)
        col1,col2=st.columns(2)
        with col1:
            #image=image.resize((150,150))
            st.image(image)
        with col2:
            if st.button('Classify'):
                ResNet=tensorflow.keras.models.load_model('ResNet50 (1).h5')
                p2=prediction(ResNet,img)
                st.success(p2)


if __name__=='__main__':
    main()
