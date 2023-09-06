# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:01:58 2023

@author: dell1
"""

import numpy as np
import pickle
import streamlit as st


loadel_model=pickle.load(open('Model/trained_model.sav', 'rb'))
loadel_model_pca=pickle.load(open('Model/trained_model_pca.sav', 'rb'))
loadel_model_scaled=pickle.load(open('Model/trained_model_scaled.sav', 'rb'))
def pred_classification(input_data):
    input_data_array = np.asarray(input_data)
    input_data_scaled = loadel_model_scaled.transform(input_data_array)
    input_data_pca = loadel_model_pca.transform(input_data_scaled)
    input_data_reshape = input_data_pca.reshape(1,-1)
    prediction = loadel_model.predict(input_data_reshape)
    
    print(prediction)
    
    if prediction[0] == 'car':
        return 'The vehicle is car'
    elif prediction[0] == 'bus':
        return 'The vehicle is bus'
    else:
        return 'The vehicle is van'
    
   
def main():
    st.title("Vehicle Classification using Silhoutte Feature")
    st.header('Enter the value of following parameter')
    compactness =st.number_input('Enter the value compactness:')
    circularity =st.number_input('Enter the value circularity:')
    distance_circularity =st.number_input('Enter the value distance_circularity:')
    radius_ratio =st.number_input('Enter the value radius_ratio:')
    pr_axis_aspect_ratio =st.number_input('Enter the value pr.axis_aspect_ratio:')
    mx_length_aspect_ratio =st.number_input('Enter the value max.length_aspect_ratio:')
    scatter_ratio =st.number_input('Enter the value scatter_ratio:')
    elongatedness =st.number_input('Enter the value elongatedness:')
    pr_axis_rectangularity =st.number_input('Enter the value pr.axis_rectangularity:')
    mx_length_rectangularity =st.number_input('Enter the value max.length_rectangularity:')
    scaled_variance =st.number_input('Enter the value scaled_variance:')
    scaled_variance_1 =st.number_input('Enter the value scaled_variance.1:')
    scaled_radius_of_gyration =st.number_input('Enter the value scaled_radius_of_gyration:')
    scaled_radius_of_gyration_1=st.number_input('Enter the value scaled_radius_of_gyration_1:')
    skewness_about =st.number_input('Enter the value skewness_about:')
    skewness_about_1 =st.number_input('Enter the value skewness_about.1:')
    skewness_about_2 =st.number_input('Enter the value skewness_about.2:')
    hollows_ratio =st.number_input('Enter the value hollows_ratio:')
    
    
    pred = ''
    if st.button('predict'):
        pred = pred_classification([[compactness,circularity,distance_circularity,radius_ratio,pr_axis_aspect_ratio,mx_length_aspect_ratio,
                     scatter_ratio,elongatedness,pr_axis_rectangularity,mx_length_rectangularity,scaled_variance,
                     scaled_variance_1,scaled_radius_of_gyration,scaled_radius_of_gyration_1,skewness_about, skewness_about_1,
                     skewness_about_2,hollows_ratio]])
       
        
    st.success(pred)
            
if __name__=='__main__':
    main()