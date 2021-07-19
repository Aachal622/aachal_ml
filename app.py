
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('logisticmodel.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('Social_Network_Ads.csv')
# Extracting independent variable:
X = dataset.iloc[:, [1,2,3]].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(sc.transform([[Gender,Age,EstimatedSalary]]))
  print("Purchased", output)
  if output==[1]:
    prediction="PERSON IS FEMALE"
  else:
    prediction="PERSON IS MALE"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:black;margin-top:10px;">Aachal Kala (PIET18CS001)</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"End term Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Decision tree classifier to predict voice is male or female")
    
    UserID = st.text_input("UserID","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender1 = st.number_input('Insert mean frequency')
    Age = st.number_input('Insert sd)
   
    EstimatedSalary = st.number_input("Insert median)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID, Gender1,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Aachal Kala")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()
