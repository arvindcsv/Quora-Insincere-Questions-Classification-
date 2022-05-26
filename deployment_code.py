# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:59:59 2022

@author: arvind
"""

import streamlit as st

import numpy as np
import pandas as pd

import joblib 

# Fxn
pipe_nb = joblib.load(open("D:/Arvind IMP/BSE Sem 2/Black Book/Streamlit/question_model.pkl", "rb"))


def predict_emotions(docx):
    results = pipe_nb.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_nb.predict_proba([docx])
    return results





def main():
    st.title("Insincere Question CLassifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home-Question in text")
        
        with st.form(key='question_clf'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1, col2 = st.columns(2)
            
            #Apply func here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))
                if (prediction == [0]):
                    st.write("Yes!! This question is Sincere :)")
                    
                else:
                    st.write("This Question Looks like an Insincere :(")
                    
                
                
                
            with col2:
                st.success("Prediction Probability")
                st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_nb.classes_)
                st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", probability]
                
                
                
                
    elif choice == "Monitor":
        st.subheader("Monitor App")
        
    else:
        st.subheader("About")
    
    


if __name__ == '__main__':
    main()
   