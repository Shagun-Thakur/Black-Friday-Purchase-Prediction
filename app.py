import streamlit as st
import numpy as np
import pickle 

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

#Mapping for better UI
occupation_dict = {f"Occupation {i}": i for i in range(21)}

# Product category options
product_cat1 = list(range(1, 20))
product_cat2 = list(range(0, 20)) 
product_cat3 = list(range(0, 20)) 

# HTML and CSS Styling
st.markdown("""
      <style>
      .main{
           background-color: #f0f2f6;
       }
      .title{
           color: #003366;
           font-size: 40px;
           font-weight: bold;
       }
       .subtitle{
             color: #333333;
             font-size: 18px;
        }
        </style>
""", unsafe_allow_html= True)

# Page title and Discription
st.markdown('<div class="title">Black Friday Customer Purchase Prediction App</div>', unsafe_allow_html=True)
st.markdown("""<div class='subtitle'>
This app predicts how much a customer is likely to spend based on inputs like
age, gender, occupation, and product category.
It uses a machine learning model trained on anonymized e-commerce data.
</div>""", unsafe_allow_html=True)

# Input fields
st.header("Enter customer details to predict the purchase amount: ")
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.selectbox("Age Group", ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
occupation_label = st.selectbox("Select Occupation", list(occupation_dict.keys()))
occupation = occupation_dict[occupation_label]
city = st.selectbox("City Category",['A', 'B', 'C'])
stay_years = st.selectbox("Stay in Current City (Years)", ['0', '1', '2', '3', '4+'])
marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])
product_category_1 = st.selectbox("Product Category 1", product_cat1)
product_category_2 = st.selectbox("Product Category 2 (optional)", product_cat2)
product_category_3 = st.selectbox("Product category 3 (optional)", product_cat3)

# Predict
if st.button("Predict Purchase Amount"):
    # Convert gender and marital_status to numeric
    gender_m = 1 if gender == 'Male' else 0
    married = 1 if marital_status == 'Married' else 0
    city_A = 1 if city == 'A' else 0
    city_B = 1 if city == 'B' else 0
    city_C = 1 if city == 'C' else 0
    # Prepare the input in correct order(10 features)
    input_data = np.array([[occupation, age, stay_years, product_category_1, product_category_2, product_category_3, gender_m, city_B, city_C, married]])

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    encoders = {'Age': LabelEncoder(), 'City_Category': LabelEncoder(), 'Stay_In_Current_City_Years': LabelEncoder()}
    input_data[:,1] = encoders['Age'].fit_transform([input_data[0][1]])
    input_data[:,3] = encoders['City_Category'].fit_transform([input_data[0][3]])
    input_data[:,4] = encoders['Stay_In_Current_City_Years'].fit_transform([input_data[0][4]])
    #Convert to float
    input_data = input_data.astype(float)

    prediction = model.predict(input_data)
    st.success(f"Predicted Purchase Amount : INR{int(prediction[0])}")

# Footer
st.markdown("---")
st.markdown("Created by **Shagun Thakur** | Connect on [LinkedIn](https://www.linkedin.com/)")