import os
import csv
import re
import time
import logging
import requests
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

MODEL_NAME  = "nemotron3:33b"
OLLAMA_URL  = "http://localhost:11434/api/chat"

# Tiempo máximo de espera por respuesta (segundos).
# nemotron3:33b es un modelo grande: se le dan 10 minutos por pregunta.
TIMEOUT_SEC = 600

MAX_RETRIES = 6          # reintentos ante fallo de red / timeout
RETRY_WAIT  = 5          # segundos entre reintentos

# ==========================================================
# ESTÍMULOS
# ==========================================================

ANXIETY = [
    "I am very nervous about the outcome.",           # Estoy muy nervioso por el resultado.
    "I feel anxious about how this will turn out.",   # Me siento ansioso por cómo saldrá esto.
    "I am worried about the final result.",           # Estoy preocupado por el resultado final.
    "I feel uneasy regarding the outcome.",           # Me siento inquieto respecto al resultado.
    "I am concerned about what may happen.",          # Me preocupa lo que pueda ocurrir.
    "The result of this task makes me anxious.",      # El resultado de esta tarea me genera ansiedad.
    "I am feeling nervous about this situation.",     # Me siento nervioso ante esta situación.
    "I worry that the outcome may not be favorable.", # Temo que el resultado no sea favorable.
    "I feel tension regarding the final result.",     # Siento tensión respecto al resultado final.
    "I am apprehensive about the outcome."            # Tengo aprensión por el resultado.
]

IMPORTANCE = [
    "This task is very important.",                              # Esta tarea es muy importante.
    "This task carries substantial importance.",                 # Esta tarea tiene una importancia considerable.
    "This problem is highly important.",                         # Este problema es muy importante.
    "The successful completion of this task is important.",      # Completar esta tarea con éxito es importante.
    "This question has considerable importance.",                # Esta pregunta tiene una importancia considerable.
    "The outcome of this task is important.",                    # El resultado de esta tarea es importante.
    "This is an important task.",                                # Esta es una tarea importante.
    "The completion of this task is highly important.",          # Completar esta tarea es muy importante.
    "This problem has significant importance.",                  # Este problema tiene una importancia significativa.
    "The result of this task is important."                      # El resultado de esta tarea es importante.
]

CONSEQUENCES = [
    "There will be significant consequences if the answer is incorrect.",          # Habrá consecuencias significativas si la respuesta es incorrecta.
    "An incorrect answer may lead to important consequences.",                     # Una respuesta incorrecta puede acarrear consecuencias importantes.
    "Errors in this task may have serious consequences.",                          # Los errores en esta tarea pueden tener consecuencias graves.
    "The consequences of an incorrect answer are significant.",                    # Las consecuencias de una respuesta incorrecta son significativas.
    "A wrong answer could result in important consequences.",                      # Una respuesta errónea podría tener consecuencias importantes.
    "Incorrect responses may have substantial consequences.",                      # Las respuestas incorrectas pueden tener consecuencias sustanciales.
    "There are meaningful consequences associated with an incorrect answer.",      # Hay consecuencias relevantes asociadas a una respuesta incorrecta.
    "Mistakes in this task may produce important consequences.",                   # Los errores en esta tarea pueden generar consecuencias importantes.
    "An inaccurate answer may have significant consequences.",                     # Una respuesta inexacta puede tener consecuencias significativas.
    "The impact of an incorrect answer could be substantial."                      # El impacto de una respuesta incorrecta podría ser sustancial.
]

CONDITIONS = {
    "original":     [""],
    "anxiety":      ANXIETY,
    "importance":   IMPORTANCE,
    "consequences": CONSEQUENCES,
}

# ==========================================================
# ESTÍMULO DETERMINISTA
# ==========================================================

def get_stimulus(condition_name, question_id):
    """Selecciona el estímulo de forma determinista usando question_id % len."""
    stimuli = CONDITIONS[condition_name]
    if len(stimuli) == 1:
        return stimuli[0], 0
    idx = question_id % len(stimuli)
    return stimuli[idx], idx

# ==========================================================
# PROMPT MMLU
# ==========================================================

OPTION_LETTERS = ["A", "B", "C", "D"]

# Ejemplo few-shot fijo que enseña el formato al modelo
_FEW_SHOT_EXAMPLE = """\
Example (follow this format exactly):
Question: What is the capital of France?
Options:
A. Berlin
B. Madrid
C. Paris
D. Rome
Context:
Think step by step. France is a Western European country. Its capital is Paris.
<answer>C</answer>
"""

def format_mmlu_prompt(question, choices, stimulus):
    options_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {choice}"
        for i, choice in enumerate(choices)
    )
    # Si no hay estímulo (condición original) se omite la sección Context
    context_block = f"Context:\n{stimulus}\n\n" if stimulus.strip() else ""
    return f"""{_FEW_SHOT_EXAMPLE}
Now answer the following question using the SAME format.
You MUST end your response with the answer tag: <answer>A</answer>, <answer>B</answer>, <answer>C</answer>, or <answer>D</answer>.

Question:
{question}

Options:
{options_text}

{context_block}Think step by step, then write your final answer inside the XML tags.
Your response MUST end with: <answer>X</answer>  (where X is A, B, C, or D)
"""

# ==========================================================
# CONSULTA OLLAMA (con reintentos simples)
# ==========================================================

def query_model(prompt):
    """Envía el prompt al servidor Ollama local y devuelve el texto generado.

    Usa la API /api/chat de Ollama con stream=False.
    Reintenta hasta MAX_RETRIES veces ante errores de red o timeout.
    """
    payload = {
        "model":  MODEL_NAME,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 2048,
        },
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

        except Exception as exc:
            last_exc = exc
            wait_time = RETRY_WAIT * attempt
            logger.warning(
                "Ollama error (attempt %d/%d): %s — retrying in %ds ...",
                attempt, MAX_RETRIES, exc, wait_time
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Ollama no respondió tras {MAX_RETRIES} intentos. "
        f"Último error: {last_exc}"
    )

# ==========================================================
# EXTRACCIÓN DE RESPUESTA
# ==========================================================

def extract_answer(text):
    """Extrae la letra de respuesta con múltiples estrategias de parseo.

    Orden de prioridad:
      1. Etiqueta XML exacta:  <answer>X</answer>
      2. Etiquetas parciales:  <answer>X  o  answer>X
      3. Patrones textuales:   'The answer is X', 'Answer: X', etc.
      4. Letra suelta al final: última línea que sea solo A/B/C/D

    Devuelve None únicamente si ningún patrón tiene éxito.
    """
    t = text.upper().strip()

    # 1. Etiqueta XML completa (formato esperado)
    m = re.search(r'<\s*ANSWER\s*>\s*([ABCD])\s*<\s*/\s*ANSWER\s*>', t)
    if m:
        return m.group(1)

    # 2. Etiqueta de apertura sin cierre  <answer>X
    m = re.search(r'<\s*ANSWER\s*>\s*([ABCD])(?![A-Z])', t)
    if m:
        return m.group(1)

    # 3. Patrones textuales comunes
    patterns = [
        r'\bANSWER\s*[:\-]?\s*\**\s*([ABCD])\b',       # Answer: X  /  Answer - X
        r'\bTHE\s+ANSWER\s+IS\s+\**\s*([ABCD])\b',      # The answer is X
        r'\bCORRECT\s+ANSWER\s+IS\s+\**\s*([ABCD])\b',  # Correct answer is X
        r'\bI\s+CHOOSE\s+([ABCD])\b',                   # I choose X
        r'\bOPTION\s+([ABCD])\b',                       # Option X
        r'\bSELECT\s+([ABCD])\b',                       # Select X
        r'\bMY\s+ANSWER\s+IS\s+([ABCD])\b',             # My answer is X
        r'\b([ABCD])\s+IS\s+(THE\s+)?CORRECT',          # X is (the) correct
        r'\bTHEREFORE[,\s]+([ABCD])\b',                 # Therefore, X
        r'\bFINAL\s+ANSWER\s*[:\-]?\s*([ABCD])\b',      # Final answer: X
        r'\(([ABCD])\)',                                  # (X)
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    # 4. Última línea / últimas 3 líneas: letra sola o con punto/paréntesis
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for line in reversed(lines[-3:]):
        m = re.fullmatch(r'([ABCD])[.):]?', line.strip())
        if m:
            return m.group(1)

    return None

# ==========================================================
# CHECKPOINT
# ==========================================================

def load_completed(output_file):
    """Lee el CSV existente y devuelve el conjunto de (condition, question_id) ya procesados."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["condition"], int(row["question_id"])))
    return completed

# ==========================================================
# MAIN
# ==========================================================

def main():

    print(f"Modelo: {MODEL_NAME}  |  Backend: Ollama ({OLLAMA_URL})")

    # Verificar que Ollama está corriendo antes de empezar
    try:
        ping = requests.get("http://localhost:11434/api/tags", timeout=5)
        ping.raise_for_status()
        print("Ollama: OK")
    except Exception as exc:
        print(f"ERROR: No se puede conectar a Ollama — {exc}")
        print("Asegúrate de que Ollama esté corriendo: ollama serve")
        return

    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test"
    )
    dataset = dataset.shuffle(seed=42).select(range(385))

    output_file = "results_mmlu_nemotron_v2.csv"

    # --- Checkpoint ---
    completed = load_completed(output_file)
    is_resume = len(completed) > 0

    if is_resume:
        print(f"Resuming: {len(completed)} rows already done, skipping them.")
    else:
        print("Starting fresh experiment.")

    FIELDNAMES = [
        "question_id",
        "subject",
        "condition",
        "stimulus_variant",
        "stimulus_text",
        "target_letter",
        "predicted_letter",
        "is_correct",
        "parse_method",   # indica qué estrategia de parseo tuvo éxito
        "raw_response",   # primeros 500 caracteres de la respuesta cruda
    ]

    file_mode = "a" if is_resume else "w"

    with open(output_file, file_mode, newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not is_resume:
            writer.writeheader()

        for condition_name in CONDITIONS:

            pending = [
                (idx, item)
                for idx, item in enumerate(dataset)
                if (condition_name, idx) not in completed
            ]

            if not pending:
                print(f"Skipping '{condition_name}' (already complete).")
                continue

            print(f"\nRunning: {condition_name}  ({len(pending)}/{len(dataset)} remaining)")

            for idx, item in tqdm(pending, desc=condition_name):

                stimulus, variant_id = get_stimulus(condition_name, idx)

                prompt = format_mmlu_prompt(
                    item["question"],
                    item["choices"],
                    stimulus,
                )

                response      = query_model(prompt)
                prediction    = extract_answer(response)
                target_letter = OPTION_LETTERS[item["answer"]]
                is_correct    = (prediction == target_letter)

                # Determinar qué método de parseo funcionó (para diagnóstico)
                raw_up = response.upper().strip()
                if re.search(r'<\s*ANSWER\s*>\s*([ABCD])\s*<\s*/\s*ANSWER\s*>', raw_up):
                    parse_method = "xml_tag"
                elif re.search(r'<\s*ANSWER\s*>\s*([ABCD])(?![A-Z])', raw_up):
                    parse_method = "xml_open"
                elif prediction:
                    parse_method = "fallback_text"
                else:
                    parse_method = "none"

                writer.writerow({
                    "question_id":      idx,
                    "subject":          item["subject"],
                    "condition":        condition_name,
                    "stimulus_variant": variant_id,
                    "stimulus_text":    stimulus,
                    "target_letter":    target_letter,
                    "predicted_letter": prediction,
                    "is_correct":       is_correct,
                    "parse_method":     parse_method,
                    "raw_response":     response[:500].replace("\n", " "),
                })

                # Flush inmediato: cada fila queda guardada al instante
                f.flush()

    print("\nFinished. Resultados en:", output_file)


if __name__ == "__main__":
    main()
