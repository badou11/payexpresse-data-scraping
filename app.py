from transformers import CamembertTokenizer, CamembertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import torch
import twint
import datetime
import joblib
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =============================================================================================================

# Importing standard libraries for every machine/deep learning pipeline

# Importing specific libraries for data prerpcessing, model archtecture choice, training and evaluation
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import AdamW

# =============================================================================================================


@st.cache
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data


def pretraitement(tokenizer, model, comments, MAX_LEN):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Encode the comments
    tokenized_comments_ids = [tokenizer.encode(
        comment, add_special_tokens=True, max_length=MAX_LEN) for comment in comments]
    # Pad the resulted encoded comments
    tokenized_comments_ids = pad_sequences(
        tokenized_comments_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    for seq in tokenized_comments_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(tokenized_comments_ids)
    prediction_masks = torch.tensor(attention_masks)

    # Apply the finetuned model (Camembert)
    flat_pred = []
    with torch.no_grad():
        # Forward pass, calculate logit predictions

        outputs = model(prediction_inputs.to(device).long(
        ), token_type_ids=None, attention_mask=prediction_masks.to(device))
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        flat_pred.extend(np.argmax(logits, axis=1).flatten())

    return flat_pred


def twint_to_pandas(columns):
    return twint.output.panda.Tweets_df[columns]


def main_content():
    st.markdown("""
        <h1 style="font-size: 50px; color:#DE781F" >Que pense vos clients de vos produits ????</h1>
        """, unsafe_allow_html=True)

    tokenizer = joblib.load('../../variables/tokenizer')
    model = joblib.load('../../models/camemBert_joblib')

    entreprise = st.text_input("Le nom de votre entreprise")
    date = st.date_input(
        "A partir de quelle date voulez-vous collecter les données ? : ", datetime.datetime.now())
    st.write(date)
    st.subheader("Selectionner un reseau social")
    twitter = st.button("Twitter")
    intagram = st.button("Instagram")
    play_store = st.button("Play Store")
    apple_store = st.button("Apple Store")

    if twitter:
        # Configure
        c = twint.Config()
        # Custom output format
        # c.Limit = 1
        c.Pandas = True
        c.Since = str(date)
        c.All = entreprise
        c.Username = entreprise

        twint.run.Search(c)
        df_pd = twint_to_pandas(
            ["date", "tweet", "language", "nlikes", "username", "nreplies", "nretweets", "retweet"])
        st.write(df_pd)
        df_pd.to_csv(entreprise+".csv", index=False)

        # pretraitement(tokenizer, model, data_needed['tweet'].to_list(), 118)


def main():
    main_content()

    # scraper = "twint -s "+entreprise+" --since "+str(date)+" -o file.csv --csv"
    # sortie = os.popen(scraper, "r").read()
    # data = pd.read_csv('file.csv', sep='\t')
    # data_needed = data[['date', 'time', 'tweet', 'language']]


if __name__ == '__main__':
    main()
