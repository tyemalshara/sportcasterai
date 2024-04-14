import cv2
from google.colab.patches import cv2_imshow
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st
# Essential custom fuctions for the app to run
from main_functions import *

# Streamlit UI elements
st.title("SportCasterAI - Soccer Goal Predictor")

uploaded_file = st.file_uploader("Upload a image file", type=["jpg", "png"])
if uploaded_file is not None:
  filename = "UserImage.jpg"
  with open(filename, "wb") as f:
    f.write(uploaded_file.getbuffer())
  st.session_state.input_file_path = "UserImage.jpg"
  # Read image
  frame = cv2.imread('UserImage.jpg')
  # display the image
  st.image(frame, caption='This is your image. It has been successfully uploaded.', channels="BGR")
  # add a spinner to the UI
  with st.spinner('Your image is being processed, results could take a few seconds to 10 minutes...'):
    # Run the workflow on current frame -> wf.run_on(array=frame)
    # Fetch the results from Ikomia's API
    response_uuid, JWT = call_IkomiaAPI()
    results = fetch_workflow_results(response_uuid, JWT)
    if results:
        print("Workflow results:", results)
        instance_segmentation_objects = results[1]['INSTANCE_SEGMENTATION']['detections']  # instance_segmentation.get_results().get_objects()
        object_detection_objects = results[0]['OBJECT_DETECTION']['detections'] # object_detection.get_results().get_objects()
        player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch = players_crossing_zones(instance_segmentation_objects, object_detection_objects, frame)
        player_with_ball = player_near_ball(object_detection_objects)
        player_team_1, player_team_2, player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch = team_detection(frame, instance_segmentation_objects)
        df = InitFrameDataDataFrame(player_id_in_crossing_zone, player_id_in_recipient_zone, player_id_in_pitch, player_with_ball, len(player_team_1), len(player_team_2), player_team_1_in_crossing_zone, player_team_1_in_recipient_zone, player_team_1_in_pitch, player_team_2_in_crossing_zone, player_team_2_in_recipient_zone, player_team_2_in_pitch, frame)

        GoalPredResults = PredictGoal(df)
        st.write(GoalPredResults)
    else:
        print("Failed to retrieve results within the given timeout.")
