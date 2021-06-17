import os

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import tempfile
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
from PIL import Image

st.set_page_config(page_title="Search for Objects in Video.", page_icon=":desktop_computer:")
st.text('Detect Objects in Video')


def save_uploaded_file(uploadedfile):
    if os.path.isfile(uploadedfile.name):
        os.remove(uploadedfile.name)
    else:
        print("Error: %s file not found" % uploadedfile.name)
    with open(os.path.join("", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved file :{}".format(uploadedfile.name))


def objects_detection(video_name):
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []
    file_name = 'Labels.txt'
    with open(file_name, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    img = cv2.imread('aston-martin-db5-7-1.jpg')

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    print(ClassIndex)

    font_scale = 4
    font = cv2.FONT_HERSHEY_PLAIN
    for ClassId, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        cv2.putText(img, classLabels[ClassId - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                    color=(0, 255, 0), thickness=3)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    font_scale = 4
    font = cv2.FONT_HERSHEY_PLAIN
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Could Not Open Video")

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    frameDetections = []
    frameLabels = []

    while True:
        ret, frame = cap.read()
        try:
            frame = cv2.resize(frame, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print("An exception occurred: ", e)
            print("error 1")
            # Release device
            cap.release()
            cv2.destroyAllWindows()
            break
        try:
            ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.68)
        except Exception as e:
            print("An exception occurred: ", e)
            print("Error 2")
            # Release device
            cap.release()
            cv2.destroyAllWindows()
            break

        print(ClassIndex)

        if len(ClassIndex) != 0:
            for ClassId, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassId <= 80:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    frame = cv2.putText(frame, classLabels[ClassId - 1], (boxes[0] + 10, boxes[1] + 40), font,
                                        fontScale=font_scale, color=(0, 255, 0), thickness=3)
                    frame_width = int(cap.get(3))
                    frame_height = int(cap.get(4))
                    frameDetections.append(frame)
                    frameLabels.append(classLabels[ClassId - 1])

        try:
            video_files = cv2.imshow('Detecting', frame)
            # st.video(video_files)
        except Exception as e:
            print("An exception occurred: ", e)
            print("Error 3")
            cap.release()
            cv2.destroyAllWindows()
            break

        # print(detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    data = {'Frames': frameDetections, 'Labels': frameLabels}
    dataFrame = pd.DataFrame(data)
    # plt.imshow(frameDetections[100])
    dataFrame.head(20)

    print(dataFrame.shape)

    print(dataFrame.iloc[5, 1])
    plt.imshow(dataFrame.iloc[5, 0])
    st.sidebar.success("All Objects Detected, Scroll Down")
    st.success("All Objects Detected, Check Summary Below")
    return dataFrame


def statics(data_frame):
    return True


def main():
    """
    #  Traffic Flow Counter :blue_car:  :red_car:
    Upload a video file to track and count vehicles. Don't forget to change parameters to tune the model!
    #### Features to be added in the future:
    + speed measurement
    + traffic density
    + vehicle type distribution
    """

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    with upload:
        video_file = st.file_uploader('Upload Video file (.mp4 format)', type=["mp4"], key=None)
    if video_file is not None:
        video_bytes = video_file.read()
        save_uploaded_file(video_file)
        st.video(video_bytes)
        st.sidebar.warning("Wait, Detecting Objects....")
        data = objects_detection(video_file.name)
        with st.form(key='searcher'):
            nav1, nav2 = st.beta_columns([3, 1])
            with nav1:
                search_term = st.text_input("Search Object")
            with nav2:
                st.text("Search ")
                submit_search = st.form_submit_button(label='Search')
            st.success("You searched for {}".format(search_term))

        # Results
        col1, col2 = st.beta_columns([2, 1])
        with col1:
            if submit_search:
                result = data.loc[data['Labels'] == search_term]
                st.subheader("Showing {} Results".format(result.Labels.size))
                st.write("The {} frame(s) with the object you searched".format(search_term))
                st.dataframe(result)
                i = 0
                while i < result.Labels.size:
                    cv2.imwrite('saved/saved{}{}.png'.format(search_term, i), result.iloc[i, 0])
                    i = i + 1
                st.write("Preview {} Results".format(search_term))
                try:
                    image = Image.open('saved/saved{}{}.png'.format(search_term, 0))
                    #image2 = Image.open('saved/saved{}{}.png'.format(search_term, 1))
                    st.text("Bar Chart of {}".format(search_term))
                    st.bar_chart(result['Labels'].value_counts())
                    st.text("Just Preview of One Image Check Saved Folder for All")
                    st.image(image)
                    #st.image(image2)
                except IOError as e:
                    st.write("No results")
                    print(e)
        st.text("Bar Chart of Object Detection")
        st.bar_chart(data['Labels'].value_counts())
        st.text("Object Detection Frequency")
        st.dataframe(data['Labels'].value_counts())
        st.text("Full Data Frame")
        st.dataframe(data)
        st.text("Full Data Frame Shape")
        st.dataframe(data.shape)

        # with st.sidebar:
        #     """
        #     # :floppy_disk: Search Frames
        #     """
        #     name = st.text_input("Search for objects in the video")


main()
