# Tema 3 - RAG Search
# Autor: Grigoras Andrei

import os
from pathlib import Path

import chromadb
import pymupdf4llm
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

# incarcam fisierele -> chunk-uri -> embeddings -> baza de date vectoriala ->
# cautare semantica doar pe fisierul cerut -> agentul raspunde din rezultate.

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_ROOT / "data_sources"
DB_FOLDER = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "tema3_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
N_RESULTS = 3
# Tema 3.1: distanta cosinus maxima acceptata. peste aceasta valoare, rezultatul e ignorat.
MAX_DISTANCE = 0.5


def creeaza_model():
    # acelasi model ca la curs (all-MiniLM-L6-v2), dar rulat de ChromaDB, fara PyTorch.
    # PyTorch nu mai are versiuni noi pentru Mac-urile cu procesor Intel.
    return DefaultEmbeddingFunction()


def embed(model, texte: list[str]) -> list[list[float]]:
    # transforma fiecare text intr-un vector (lista de numere). vectorii sunt deja normalizati.
    vectori = []
    for vector in model(texte):
        vectori.append([float(x) for x in vector])
    return vectori


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def citeste_pagini(file_path: Path) -> list[tuple[int, str]]:
    # returneaza o lista de (numar_pagina, text)
    if file_path.suffix.lower() == ".pdf":
        pagini = []
        for page in pymupdf4llm.to_markdown(str(file_path), page_chunks=True):
            pagini.append((page["metadata"]["page_number"], page["text"]))
        return pagini

    # fisierele text nu au pagini, le consideram o singura pagina
    return [(1, file_path.read_text(encoding="utf-8"))]


def index_file(collection, model, file_path: Path) -> int:
    ids = []
    documents = []
    metadatas = []

    for page_number, page_text in citeste_pagini(file_path):
        chunk_number = 0
        for chunk in chunk_text(page_text):
            ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
            documents.append(chunk)
            # "source" = numele fisierului, ca sa putem filtra dupa el la cautare
            metadatas.append({
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number,
            })
            chunk_number += 1

    # stergem chunk-urile vechi ale fisierului, ca sa nu avem duplicate
    collection.delete(where={"source": file_path.name})

    if len(documents) == 0:
        return 0

    collection.upsert(
        ids=ids,
        embeddings=embed(model, documents),
        documents=documents,
        metadatas=metadatas,
    )
    return len(documents)


def index_folder(collection, model, folder: Path = DATA_FOLDER) -> list[str]:
    surse = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file():
            count = index_file(collection, model, file_path)
            print(f"ingest> {file_path.name}: {count} chunk-uri")
            surse.append(file_path.name)
    return surse


def are_date(collection, source: str) -> bool:
    # verificam daca exista macar un chunk din acel fisier in baza de date
    result = collection.get(where={"source": source}, limit=1)
    return len(result["ids"]) > 0


def cauta(collection, model, source: str, intrebare: str) -> list[str]:
    query_embedding = embed(model, [intrebare])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
        where={"source": source},
        include=["documents", "metadatas", "distances"],
    )

    rezultate = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):
        metadata = metadatas[i]
        distance = distances[i]

        # distanta cosinus: 0 = identic, cu cat e mai mare, cu atat e mai putin relevant.
        # pastram doar rezultatele sub limita MAX_DISTANCE.
        if distance > MAX_DISTANCE:
            print(f"search> {metadata['source']} p{metadata['page']} distance={distance:.2f} (ignorat)")
            continue

        print(f"search> {metadata['source']} p{metadata['page']} distance={distance:.2f} (pastrat)")
        rezultate.append(
            f"{documents[i]}\nSource: {metadata['source']}, page {metadata['page']}"
        )
    return rezultate


def raspunde(client: OpenAI, intrebare: str, rezultate: list[str]) -> str:
    context = "\n\n".join(rezultate)

    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=[
            {
                "role": "system",
                "content": "Answer only from the supplied context. If it does not answer the question, say so. "
                           "For every answer, mention the file name and the page number from the Source lines.",
            },
            {"role": "user", "content": f"Question: {intrebare}\n\nContext:\n{context}"},
        ],
    ).choices[0].message.content


def citeste_input(text: str) -> tuple[str, str]:
    # "paragraphs.txt, Despre cine e vorba?" -> ("paragraphs.txt", "Despre cine e vorba?")
    # split doar la prima virgula, pentru ca intrebarea poate contine si ea virgule
    parti = text.split(",", 1)
    if len(parti) != 2:
        return "", ""
    return parti[0].strip(), parti[1].strip()


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    model = creeaza_model()
    collection = chromadb.PersistentClient(path=str(DB_FOLDER)).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    surse = index_folder(collection, model)
    print(f"Fisiere disponibile: {', '.join(surse)}")
    print("Scrie: nume_fisier, intrebare  (sau quit)")

    while True:
        text = input("you> ")
        if text == "quit":
            break

        fisier, intrebare = citeste_input(text)
        if not fisier or not intrebare:
            print("agent> Format gresit. Exemplu: paragraphs.txt, Despre cine e vorba?")
            continue

        if not are_date(collection, fisier):
            print(f"agent> Nu am date despre fisierul '{fisier}'.")
            continue

        rezultate = cauta(collection, model, fisier, intrebare)

        # toate rezultatele au fost prea departe de intrebare -> nu mai intrebam agentul
        if len(rezultate) == 0:
            print(f"agent> Nu s-a gasit niciun rezultat relevant in '{fisier}' (distanta peste {MAX_DISTANCE}).")
            continue

        print("agent>", raspunde(client, intrebare, rezultate))


if __name__ == "__main__":
    main()
