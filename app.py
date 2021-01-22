import streamlit as st
import pandas as pd 
import seaborn as sns
sns.set_style('whitegrid')
import joblib
import numpy as np
import cv2

#personalise

st.markdown(
    """
    <style>
     .main {
     background-color: #F5F5F5;

     }

    </style>
    """,
      unsafe_allow_html=True
  )



#create containers
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

#write in header section
with header:
    st.title("Welcome to My Project Page")
    st.text('This Project is an ML Model for IRIS dataset')

with dataset:
    st.header('IRIS Dataset')
    st.text('This Dataset is available in open domain...')
    data = pd.read_csv('data/iris.csv')#read the csv file into df
    st.write(data.head(3)) # output first 10 lines of dataset
    st.subheader('Count plot of Species Column in Dataset')
    st.bar_chart(data['Species'].value_counts(),height=300,width=200)
    

with features:
    st.header('Dataset Features')
    st.markdown('* **The features of this dataset are** ')
    st.write(data.columns)

with model_training:
    st.header('Model Prediction')
    st.subheader('Model Predicts Species type based on user input of features')
    sel_col,disp_col = st.beta_columns(2)
    sepal_length = sel_col.slider('Input Sepal Length',0,8,2)
    sepal_width = sel_col.slider('Input Sepal Width',0,8,2)
    petal_length = sel_col.slider('Input Petal Length',0,6,2)
    petal_width = sel_col.slider('Input  Length',0.0,2.0,0.2)

    #using a drop boz to display an image of flower
    image = sel_col.selectbox('Watch the Flower',options=['Setosa','Versicolor','Virginica'])# drop down 
    uploaded_file = st.file_uploader("/images/setosa", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")



    #get input data
    disp_col.subheader('The input values entered by User')
    disp_col.write('Sepal Length')
    disp_col.write(sepal_length)
    disp_col.write('sepal_width')
    disp_col.write(sepal_width)
    disp_col.write('Petal Length')
    disp_col.write(petal_length)
    disp_col.write('Petal Width')
    disp_col.write(petal_width)
    # Load the model from the file 
logreg_from_joblib = joblib.load('iris.pkl')  

X_test = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

# Use the loaded model to make predictions 
result = logreg_from_joblib.predict(X_test)
if result == 0:
    answer = 'Setosa'
elif result==1:
    answer = 'Versicolor'
else:
    answer = 'Virginica'

st.subheader('The IRIS Flower for given input is ..')
st.write(answer)