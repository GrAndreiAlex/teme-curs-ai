import chromadb
import pytest

from tema3_rag import are_date, creeaza_model, cauta, citeste_input, index_folder


# fixture: se ruleaza o singura data pentru toate testele din fisier.
# folosim o baza de date in memorie, ca testele sa nu atinga chroma_db.
@pytest.fixture(scope="module")
def db():
    model = creeaza_model()
    collection = chromadb.EphemeralClient().get_or_create_collection(
        name="test_tema3",
        metadata={"hnsw:space": "cosine"},
    )
    index_folder(collection, model)
    return collection, model


def test_cautare_doar_in_sursa_ceruta(db):
    collection, model = db
    rezultate = cauta(collection, model, "paragraphs.txt", "Who is the text about?")

    assert len(rezultate) > 0
    for rezultat in rezultate:
        assert "Source: paragraphs.txt" in rezultat
        assert "StarMaker.pdf" not in rezultat


def test_fisier_existent_si_inexistent(db):
    collection, model = db

    assert are_date(collection, "StarMaker.pdf")
    assert not are_date(collection, "nu_exista.pdf")


def test_citeste_input():
    fisier, intrebare = citeste_input("paragraphs.txt, Despre cine e vorba, in text?")

    assert fisier == "paragraphs.txt"
    assert intrebare == "Despre cine e vorba, in text?"
