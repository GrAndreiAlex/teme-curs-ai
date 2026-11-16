# Tema 1 - Citire JSON si validare Pydantic
# Autor: Grigoras Andrei

import json
from pathlib import Path

from pydantic import BaseModel, EmailStr, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent


class StudentValidator(BaseModel):
    nume: str
    # gt=0 -> varsta trebuie sa fie mai mare ca 0
    varsta: int = Field(gt=0)
    # EmailStr verifica daca email-ul este valid
    email: EmailStr


def citeste_student(nume_fisier="student.json"):
    with open(PROJECT_ROOT / nume_fisier, "r") as f:
        # json.load() transforma continutul fisierului JSON intr-un dict
        data = json.load(f)
    return data


def valideaza_student(data):
    return StudentValidator.model_validate(data, strict=True)


if __name__ == "__main__":
    data = citeste_student()
    print(data)

    try:
        student = valideaza_student(data)
        print("Studentul este valid!")
        print(student)
    except ValidationError as e:
        print("Date invalide:")
        print(e)
