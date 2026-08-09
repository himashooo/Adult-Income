import streamlit as st
import pandas as pd
import joblib

model = joblib.load("annual_income.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💰 Annual Income Prediction")
st.write("Predict whether annual income is >50K or <=50K")

st.divider()

age = st.number_input(
    "Age",
    min_value=17,
    max_value=100,
    value=30
)

education_num = st.number_input(
    "Education Number",
    min_value=1,
    max_value=16,
    value=10
)

capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    value=0
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    value=0
)

hours_per_week = st.number_input(
    "Hours per Week",
    min_value=1,
    max_value=100,
    value=40
)

workclass = st.selectbox(
    "Workclass",
    [
        "Federal-gov",
        "Local-gov",
        "Never-worked",
        "Private",
        "Self-emp-inc",
        "Self-emp-not-inc",
        "State-gov",
        "Without-pay"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Divorced",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Never-married",
        "Separated",
        "Widowed"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving"
    ]
)

relationship = st.selectbox(
    "Relationship",
    [
        "Not-in-family",
        "Other-relative",
        "Own-child",
        "Unmarried",
        "Wife"
    ]
)

sex = st.selectbox(
    "Sex",
    [
        "Male",
        "Female"
    ]
)

race = st.selectbox(
    "Race",
    [
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White"
    ]
)

native_country = st.selectbox(
    "Native Country",
    [
        "Canada",
        "China",
        "Columbia",
        "Cuba",
        "Dominican-Republic",
        "Ecuador",
        "El-Salvador",
        "England",
        "France",
        "Germany",
        "Greece",
        "Guatemala",
        "Haiti",
        "Holand-Netherlands",
        "Honduras",
        "Hong",
        "Hungary",
        "India",
        "Iran",
        "Ireland",
        "Italy",
        "Jamaica",
        "Japan",
        "Laos",
        "Mexico",
        "Nicaragua",
        "Outlying-US(Guam-USVI-etc)",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Puerto-Rico",
        "Scotland",
        "South",
        "Taiwan",
        "Thailand",
        "Trinadad&Tobago",
        "United-States",
        "Vietnam",
        "Yugoslavia"
    ]
)


if st.button("Predict Income"):

    data = pd.DataFrame({
        "age": [age],
        "education_num": [education_num],
        "capital_gain": [capital_gain],
        "capital_loss": [capital_loss],
        "hours_per_week": [hours_per_week],
        "workclass": [workclass],
        "marital_status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "sex": [sex],
        "race": [race],
        "native_country": [native_country]
    })

    data = pd.get_dummies(
        data,
        columns=[
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "sex",
            "race",
            "native_country"
        ],
        drop_first=True
    )

    data = data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        st.success("💰 Income is likely to be **>50K**")
    else:
        st.info("Income is likely to be **<=50K**")