# Copyright (c) 2026 Hamza Zaroual
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.


import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
import openai

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv

# Chargement .env
load_dotenv()

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class FraudRAGAgent:
    """
    RAG Agent sécurisé :
    - Recherche FAISS sur transactions
    - Réponses toujours basées sur les documents retrouvés
    - GPT-3.5 via OpenAI si disponible, sinon fallback local
    """

    def __init__(self, max_transactions=100, n_jobs=4, save_dir="faiss_data"):
        self.max_transactions = max_transactions
        self.n_jobs = n_jobs
        self.device = "cpu"
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.embeddings_file = os.path.join(self.save_dir, "embeddings.npy")
        self.index_file = os.path.join(self.save_dir, "faiss.index")

        print("Initialisation du FraudRAGAgent")

        # Embeddings
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )

        # OpenAI config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = (
            OPENAI_AVAILABLE and
            self.openai_api_key is not None
        )

        if self.use_openai:
            openai.api_key = self.openai_api_key
            print("OpenAI GPT-3.5 activé")
        else:
            print("OpenAI indisponible, utilisation du modèle local")

        # LLM local (fallback)
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.local_llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )

        self.documents = []
        self.embeddings = None
        self.index = None

    # ------------------------------------------------------------------
    def load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.iloc[: self.max_transactions]
        print(f"{len(df)} transactions chargées")
        return df

    # ------------------------------------------------------------------
    def _make_document(self, idx, row):
        text = (
            f"Transaction {idx} | "
            f"Amount: {row['Amount']} € | "
            f"Fraud score: {row['fraud_final_proba']:.5f} | "
            f"Label: {'FRAUD' if row['fraud_final_label'] == 1 else 'LEGIT'}"
        )
        return {
            "id": idx,
            "text": text,
            "metadata": row.to_dict()
        }

    # ------------------------------------------------------------------
    def build_documents(self, df: pd.DataFrame):
        self.documents = Parallel(n_jobs=self.n_jobs)(
            delayed(self._make_document)(i, r)
            for i, r in df.iterrows()
        )

    # ------------------------------------------------------------------
    def build_faiss_index(self, rebuild=False):
        if (
            not rebuild
            and os.path.exists(self.embeddings_file)
            and os.path.exists(self.index_file)
        ):
            print("Chargement de l'index FAISS existant")
            self.embeddings = np.load(self.embeddings_file)
            self.index = faiss.read_index(self.index_file)
            return

        print("Calcul des embeddings")
        texts = [d["text"] for d in self.documents]

        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

        np.save(self.embeddings_file, self.embeddings)
        faiss.write_index(self.index, self.index_file)

        print("Index FAISS prêt")

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5):
        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        _, indices = self.index.search(
            query_emb.astype("float32"),
            k
        )

        return [self.documents[i] for i in indices[0]]

    # ------------------------------------------------------------------
    def _openai_generate(self, prompt: str) -> str | None:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial fraud assistant. "
                            "Use only the provided transactions. "
                            "If the answer is not in the data, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=200,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print("Erreur OpenAI :", e)
            return None

    # ------------------------------------------------------------------
    def generate_answer(self, query: str, retrieved_docs: List[dict]) -> str:
        if not retrieved_docs:
            return "No relevant transactions found."

        context = "\n".join(d["text"] for d in retrieved_docs)

        prompt = f"""
Transactions:
{context}

User question:
{query}

Answer strictly based on the transactions above:
"""

        if self.use_openai:
            answer = self._openai_generate(prompt)
            if answer:
                return answer

        output = self.local_llm(
            prompt,
            max_new_tokens=150,
            temperature=0.5,
            do_sample=False
        )

        text = output[0]["generated_text"][len(prompt):].strip()
        return text if text else context

    # ------------------------------------------------------------------
    def ask(self, query: str, k: int = 5):
        retrieved = self.retrieve(query, k)
        answer = self.generate_answer(query, retrieved)
        return {
            "query": query,
            "retrieved_transactions": retrieved,
            "answer": answer,
        }


# ======================================================================
def main():
    csv_path = "../results/final_predictions.csv"

    agent = FraudRAGAgent(max_transactions=100)
    df = agent.load_data(csv_path)
    agent.build_documents(df)
    agent.build_faiss_index()

    query = "Show transactions with high fraud risk"
    result = agent.ask(query, k=5)

    print("\nQUERY:")
    print(result["query"])

    print("\nTRANSACTIONS:")
    for d in result["retrieved_transactions"]:
        print("-", d["text"])

    print("\nANSWER:")
    print(result["answer"])


if __name__ == "__main__":
    main()
