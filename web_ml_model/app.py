#We import all neccessary libraries for our project
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

pickle_in=open("model.pkl","rb")
classifier=pickle.load(pickle_in)



def main():
#Title of our web application
    st.title("Alzheimer_Status prediction app")
    #template for header of web page
    html_temp="""
       <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Alzheimer_Status prediction ML App </h2>
    </div>
      """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Collect user input for prediction
    age = st.text_input("Enter Age", "70")
    memory_score = st.text_input("Enter Memory Score (between 0 and 1)", "0.8")
    brain_size = st.text_input("Enter Brain Size", "1200")
   #we make a predict button ,on clicking it our predict_alzheimer_status function triggers and helps us to predict our desired output
    if st.button("predict"):
        prediction=predict_alzheimer_status(classifier,float(age),float(memory_score),float(brain_size))
        st.subheader("prediction")
        #printing our resullt of prediction
        if prediction == 1:
            st.write("The model predicts Alzheimer's disease.")
        else:
            st.write("The model predicts no Alzheimer's disease.")

#defining predict_alzheimer_status fuction's functionalities
def predict_alzheimer_status(classifier,age,memory_score,brain_size):
    input_data = pd.DataFrame({'Age': [age], 'Memory_Score': [memory_score], 'Brain_Size': [brain_size]})
    prediction = classifier.predict(input_data)
    return prediction[0]





#main function call
if __name__=='__main__':
    main()