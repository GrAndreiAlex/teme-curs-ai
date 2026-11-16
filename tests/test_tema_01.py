import pytest
from pydantic import ValidationError

from tema_01_json_pydantic import StudentValidator, citeste_student, valideaza_student


def test_student_valid():
    data = citeste_student()
    student = valideaza_student(data)

    assert isinstance(student, StudentValidator)
    assert student.nume == "Ana Popescu"
    assert student.varsta == 20
    assert student.email == "ana.popescu@example.com"


@pytest.mark.parametrize("email", ["ana.popescu", "ana@", "@example.com", ""])
def test_email_invalid(email):
    data = citeste_student()
    data["email"] = email

    with pytest.raises(ValidationError):
        valideaza_student(data)


@pytest.mark.parametrize("varsta", [0, -1, -20])
def test_varsta_invalida(varsta):
    data = citeste_student()
    data["varsta"] = varsta

    with pytest.raises(ValidationError):
        valideaza_student(data)


def test_varsta_nu_e_numar():
    data = citeste_student()
    data["varsta"] = "20"

    with pytest.raises(ValidationError):
        valideaza_student(data)


def test_nume_invalid():
    data = citeste_student()
    data["nume"] = 10

    with pytest.raises(ValidationError):
        valideaza_student(data)
