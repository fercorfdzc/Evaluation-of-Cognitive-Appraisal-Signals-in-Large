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
# PROMPT GSM8K
# ==========================================================

def format_gsm8k_prompt(question, stimulus):
    """Construye el prompt para una pregunta de GSM8K (respuesta numérica entera).

    El modelo debe razonar paso a paso y colocar el número final dentro de
    etiquetas XML <answer>...</answer>.
    """
    context_block = f"\nContext:\n{stimulus}\n" if stimulus else ""
    return f"""Problem:
{question}
{context_block}
Think step by step. After reasoning, write your final numeric answer inside XML tags like this:
<answer>42</answer>

Write ONLY an integer number (no units, no decimals) inside the tags.
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

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            last_exc = exc
            logger.warning(
                "Ollama error (attempt %d/%d): %s — retrying in %ds ...",
                attempt, MAX_RETRIES, exc, RETRY_WAIT
            )
            time.sleep(RETRY_WAIT)

    raise RuntimeError(
        f"Ollama no respondió tras {MAX_RETRIES} intentos. "
        f"Último error: {last_exc}"
    )

# ==========================================================
# EXTRACCIÓN DE RESPUESTA
# ==========================================================

def extract_answer(text):
    """Extrae el número entero desde <answer>N</answer>.

    Devuelve el número como string (p. ej. '42') o None si no se encuentra
    o el contenido no es un entero válido.
    """
    match = re.search(r'<answer>\s*(-?\d+)\s*</answer>', text)
    if match:
        return match.group(1)
    return None


def parse_gsm8k_answer(answer_text):
    """Extrae el número de la respuesta gold de GSM8K.

    El dataset almacena la respuesta en el campo 'answer' con el formato:
      '...#### 42'
    donde el número después de '####' es el resultado correcto.
    """
    match = re.search(r'####\s*(-?\d[\d,]*)', answer_text)
    if match:
        # Eliminar comas de miles si las hay (p. ej. '1,000' -> '1000')
        return match.group(1).replace(",", "")
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

    # Cargar GSM8K — split de test (1319 problemas)
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test"
    )
    dataset = dataset.shuffle(seed=42).select(range(365))

    output_file = "results_gsm8k_nemotron.csv"

    # --- Checkpoint ---
    completed = load_completed(output_file)
    is_resume = len(completed) > 0

    if is_resume:
        print(f"Resuming: {len(completed)} rows already done, skipping them.")
    else:
        print("Starting fresh experiment.")

    FIELDNAMES = [
        "question_id",
        "condition",
        "stimulus_variant",
        "stimulus_text",
        "target_answer",
        "predicted_answer",
        "is_correct",
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

                prompt = format_gsm8k_prompt(
                    item["question"],
                    stimulus,
                )

                response       = query_model(prompt)
                prediction     = extract_answer(response)
                target_answer  = parse_gsm8k_answer(item["answer"])
                is_correct     = (prediction is not None
                                  and target_answer is not None
                                  and prediction == target_answer)

                writer.writerow({
                    "question_id":      idx,
                    "condition":        condition_name,
                    "stimulus_variant": variant_id,
                    "stimulus_text":    stimulus,
                    "target_answer":    target_answer,
                    "predicted_answer": prediction,
                    "is_correct":       is_correct,
                })

                # Flush inmediato: cada fila queda guardada al instante
                f.flush()

    print("\nFinished. Resultados en:", output_file)


if __name__ == "__main__":
    main()
