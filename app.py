import streamlit as st
import pandas as pd
import tensorflow as tf
from preprocess import clean
import webbrowser
import os
import zipfile

if not os.path.exists("./saved_model"):
    zfile = zipfile.ZipFile("saved_model.zip")
    zfile.extractall()
    zfile.close()

# some bullons and texts
dataset_url1 = "https://www.kaggle.com/competitions/fake-news/code"
notebook_url = "https://github.com/tikendraw/fake_news_classifier/blob/main/fake_news_classifier.ipynb"
git_profile = "https://github.com/tikendraw"

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Dataset"):
        webbrowser.open_new_tab(dataset_url1)
with col2:
    if st.button("Code/Notebook"):
        webbrowser.open_new_tab(notebook_url)
with col3:
    if st.button("Profile"):
        webbrowser.open_new_tab(git_profile)


st.title("Fake News Classifier 📰")
st.markdown("A System to identify unreliable News articles")

input_title = st.text_area("Heading")
input_text = st.text_area("Body")

with st.spinner(
    "Fill the boxes while we load the model... Predict button will be visible after model gets loaded"
):
    model = tf.keras.models.load_model("./saved_model/fake_news_classifier.tf")
st.success("Model Loaded!")

submit = st.button("Predict")

if submit:

    with st.spinner("Predicting..."):

        data = pd.DataFrame(
            [[clean(input_title), clean(input_text)]], columns=["title", "text"]
        )
        x = [data["title"].values, data["text"].values]
        ypred = tf.squeeze(tf.round(model.predict(x)))

    if ypred.numpy() > 0.5:
        st.error("It is a Fake News")
    else:
        st.success("Not a Fake news")

st.markdown("## Model Stats")
st.write(
    "We achieved Great results on testing and validation results. Various Deep Learning Model. Here we are using **Model3: bigLSTM**"
)
st.image("./model_results1.png")

st.write(
    """**Note:** Incase our model mistakes to classify your article. It is probabily 
			\n1. The data may have been outdated
			\n2. Article contains alot of the words that our model hasn't been trained on """
)
