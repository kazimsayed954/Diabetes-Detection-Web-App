import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create a title and subtitle
st.write("""
# Diabetes Detection

Detect if someone has diabetes using machine learning and python
	""")

#Open and display an Image

image=Image.open("./image/diabetesDetection.jpg")
st.image(image,caption="ML",use_column_width=True)

#Get Data
df=pd.read_csv("./data/diabetes.csv")

#Set a subheader
st.subheader('Data Information:')

#Show data as table
st.dataframe(df)

#Show statistics on Data
st.write(df.describe())

#Show the data as a Chart
chart=st.bar_chart(df)

#Split the data into Independent 'X' and Dependent 'Y' variables
X=df.iloc[:, 0:8].values
Y=df.iloc[:, -1].values

#Split the data into 75% training and 25% testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#Get the feature input from the user
def get_user_input():
	pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
	glucose=st.sidebar.slider('Glucose',0,199,117)
	blood_pressure=st.sidebar.slider('BloodPressure',0,122,77)
	skin_thickness=st.sidebar.slider('SkinThickness',0,99,23)
	insulin=st.sidebar.slider('Insulin',0.0,846.0,30.0)
	BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
	DPF=st.sidebar.slider('DPF (DiabetesPedigreeFunction)',0.078,2.42,0.3725) #DiabetesPedigreeFunction(DPF)
	age=st.sidebar.slider('Age',21,81,29)

	#Store a dictionary into a variable
	user_data={
	'Pregnancies':pregnancies,
	'Glucose':glucose,
	'BloodPressure':blood_pressure,
	'SkinThickness': skin_thickness,
	'Insulin':insulin,
	'BMI':BMI,
	'DiabetesPedigreeFunction (DPF)':DPF,
	'Age':age
	}

	#Transform the data into a data frame
	features=pd.DataFrame(user_data,index=[0])
	return features

#Store a user input into a variable
user_input=get_user_input()

#Set a subheader and dispaly the user input
st.subheader('User Input:')
st.write(user_input)

#Create and train the Model
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show the model matrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100 )+ " %")

#Store the model prediction in a variable
prediction=RandomForestClassifier.predict(user_input)

#Set a subheader and display the classification
st.subheader('Classification')
st.write(prediction)

