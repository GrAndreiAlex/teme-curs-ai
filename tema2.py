# Tema 2 - Agent simplu pentru cautare in proiect
# Autor: Grigoras Andrei

import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent
CUVANT_CHEIE = "pytest"
MAX_PASI = 5

# foldere pe care nu vrem sa le cautam (librarii instalate, git, setari IDE)
FOLDERE_IGNORATE = [".venv", ".git", ".idea", "__pycache__", ".pytest_cache"]


class CautareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(description="Cuvantul cheie cautat in continutul fisierelor.")


def este_ignorat(cale: str) -> bool:
    for folder in FOLDERE_IGNORATE:
        if folder in cale:
            return True
    return False


def ruleaza_comanda(comanda: list[str]) -> str:
    result = subprocess.run(
        comanda,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout


def filtreaza(output: str) -> list[str]:
    fisiere = []
    for linie in output.splitlines():
        if linie.strip() and not este_ignorat(linie):
            fisiere.append(linie.strip())
    return fisiere


def listeaza_fisiere() -> str:
    # find . -type f -> toate fisierele din proiect, din toate folderele
    output = ruleaza_comanda(["find", ".", "-type", "f"])
    return "\n".join(filtreaza(output))


def cauta_cuvant(args: dict) -> str:
    try:
        request = CautareRequest.model_validate(args)
    except ValidationError as e:
        return f"error: {e}"

    # grep -r -> cauta in toate folderele, -i -> nu tine cont de litere mari/mici,
    # -l -> afiseaza doar numele fisierelor in care s-a gasit cuvantul
    output = ruleaza_comanda(["grep", "-r", "-i", "-l", request.keyword, "."])
    fisiere = filtreaza(output)

    if len(fisiere) == 0:
        return f"Cuvantul '{request.keyword}' nu a fost gasit in niciun fisier."
    return "\n".join(fisiere)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "listeaza_fisiere",
            "description": "Listeaza toate fisierele din proiect."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cauta_cuvant",
            "description": "Cauta un cuvant cheie in continutul tuturor fisierelor din proiect si returneaza fisierele in care a fost gasit.",
            "parameters": CautareRequest.model_json_schema()
        }
    }
]

SYSTEM_PROMPT = """
Esti un asistent care cauta informatii intr-un proiect.
Pasii pe care trebuie sa ii urmezi:
1. Apeleaza mai intai unealta listeaza_fisiere, ca sa vezi ce fisiere are proiectul.
2. Apoi apeleaza unealta cauta_cuvant cu cuvantul cheie primit de la utilizator.
3. La final, raspunde cu lista fisierelor in care a fost gasit cuvantul cheie, exact cum le-a returnat unealta.
Nu inventa fisiere. Foloseste doar ce returneaza uneltele.
"""


def complete(client: OpenAI, messages: list[dict]):
    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    ).choices[0].message


def ruleaza_agent(client: OpenAI, keyword: str = CUVANT_CHEIE) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Cauta cuvantul '{keyword}' in tot proiectul."}
    ]

    response = complete(client, messages)
    messages.append(response.model_dump(exclude_none=True))

    # agentul cere uneltele una dupa alta (intai listare, apoi cautare).
    # cat timp cere unelte, le rulam si ii dam rezultatul inapoi.
    pas = 0
    while response.tool_calls and pas < MAX_PASI:
        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            print(f"tool> agentul a apelat: {tool_name}")

            if tool_name == "listeaza_fisiere":
                output = listeaza_fisiere()
            elif tool_name == "cauta_cuvant":
                args = json.loads(tool_call.function.arguments)
                output = cauta_cuvant(args)
            else:
                output = f"error: unealta necunoscuta {tool_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output
            })

        response = complete(client, messages)
        messages.append(response.model_dump(exclude_none=True))
        pas += 1

    return response.content or ""


def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    raspuns = ruleaza_agent(client)
    print("agent> Fisierele care contin cuvantul 'pytest':")
    print(raspuns)


if __name__ == "__main__":
    main()
