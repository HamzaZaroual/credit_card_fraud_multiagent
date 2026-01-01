# Copyright (c) 2026 Hamza Zaroual
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.



import streamlit as st
import pandas as pd
import os

from agents.RAG_agent import FraudRAGAgent

st.set_page_config(
    page_title="Fraud RAG Assistant",
    layout="wide"
)

st.title("Fraud Detection RAG Assistant")
st.write("Posez une question sur les transactions. Les réponses sont basées uniquement sur les données chargées.")

# ---------------------------------------------------------
# Lire le CSV pour connaître le nombre total de transactions
# ---------------------------------------------------------
csv_path = "./results/final_predictions.csv"
df_full = pd.read_csv(csv_path)
total_transactions = len(df_full)

st.subheader(f"Nombre total de transactions dans le CSV : {total_transactions}")

# ---------------------------------------------------------
# Saisie de l'utilisateur pour limiter le nombre de transactions
# ---------------------------------------------------------
max_transactions = st.number_input(
    "Nombre de transactions à charger (max)",
    min_value=1,
    max_value=total_transactions,
    value=min(100, total_transactions),
    step=1
)

# ---------------------------------------------------------
# Chargement et mise en cache de l'agent
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_agent(max_transactions: int):
    agent = FraudRAGAgent(max_transactions=max_transactions)

    df = agent.load_data(csv_path)
    agent.build_documents(df)
    agent.build_faiss_index()

    return agent

agent = load_agent(max_transactions)

# ---------------------------------------------------------
# Interface utilisateur pour la query
# ---------------------------------------------------------
st.subheader("Entrer votre requête")

query = st.text_input(
    "Question",
    placeholder="Ex: Show transactions with high fraud risk"
)

k = st.slider(
    "Nombre de transactions à récupérer pour la réponse",
    min_value=1,
    max_value=10,
    value=5
)

ask_button = st.button("Analyser")

# ---------------------------------------------------------
# Traitement de la requête
# ---------------------------------------------------------
if ask_button and query.strip():
    with st.spinner("Recherche des transactions pertinentes..."):
        result = agent.ask(query, k=k)

    st.subheader("Transactions récupérées")
    for d in result["retrieved_transactions"]:
        st.markdown(
            f"**Transaction ID:** {d['id']}  \n{d['text']}"
        )

    st.subheader("Réponse du modèle")
    st.success(result["answer"])

# ---------------------------------------------------------
# Bouton nouvelle requête
# ---------------------------------------------------------
st.divider()
if st.button("Nouvelle requête"):
    st.experimental_rerun()
