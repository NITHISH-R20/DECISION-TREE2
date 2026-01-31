import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder



st.set_page_config(page_title="Tic-Tac-Toe Prediction", layout="centered")

st.title("🎮 Tic-Tac-Toe Win Prediction")
st.write("Decision Tree Classifier")



df = pd.read_csv("tic-tac-toe.csv")


encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]



model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(X, y)

st.success("Model trained successfully!")

 
 
st.subheader("Enter Board Positions")
st.write("Use: **x**, **o**, **b (blank)**")

options = ["x", "o", "b"]

col1, col2, col3 = st.columns(3)

with col1:
    tl = st.selectbox("Top Left", options)
    ml = st.selectbox("Middle Left", options)
    bl = st.selectbox("Bottom Left", options)

with col2:
    tm = st.selectbox("Top Middle", options)
    mm = st.selectbox("Middle Middle", options)
    bm = st.selectbox("Bottom Middle", options)

with col3:
    tr = st.selectbox("Top Right", options)
    mr = st.selectbox("Middle Right", options)
    br = st.selectbox("Bottom Right", options)

 
 
if st.button("Predict Result"):
    mapping = {"x": 2, "o": 1, "b": 0}

    user_data = [
        mapping[tl], mapping[tm], mapping[tr],
        mapping[ml], mapping[mm], mapping[mr],
        mapping[bl], mapping[bm], mapping[br]
    ]

    result = model.predict([user_data])[0]

    if result == 1:
        st.success("🎯 Prediction: **Positive (Winning Position)**")
    else:
        st.error("❌ Prediction: **Negative (Losing Position)**")
