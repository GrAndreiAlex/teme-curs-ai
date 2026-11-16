from tema2 import cauta_cuvant, listeaza_fisiere


def test_listeaza_fisiere():
    fisiere = listeaza_fisiere()

    assert "tema2.py" in fisiere
    assert "pyproject.toml" in fisiere
    assert ".venv" not in fisiere


def test_cauta_cuvant_pytest():
    rezultat = cauta_cuvant({"keyword": "pytest"})

    # pytest.ini si pyproject.toml contin sigur cuvantul "pytest"
    assert "pytest.ini" in rezultat
    assert "pyproject.toml" in rezultat


def test_cauta_cuvant_inexistent():
    rezultat = cauta_cuvant({"keyword": "cuvant_care_nu_exista_" + "12345"})

    assert "nu a fost gasit" in rezultat
