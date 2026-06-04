import numpy as np 
import pandas as pd 
import requests 
import re 

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

def query_model(prompt, model_name, max_tokens, client, system_prompt=""):
    """Send one prompt to the proxy and return the model's response.

    Parameters
    ----------
    prompt        : str   the user prompt (the question or batch of questions)
    model_name    : str   "llama-1b", or "llama-8b" 
    max_tokens    : int   how many tokens to generate
    client        : str   proxy URL from setup_client()
    system_prompt : str   optional system instruction

    Returns
    -------
    dict with keys:
        "answer"      : str   the generated text
        "logprobs"    : dict  {token: logprob}
        "token_probs" : dict  {token: probability}
        "logprob_contents" : list of dictionaries for each generated token[{}]
    or None if the request was rate-limited / rejected.
    """
    response = requests.post(
        f"{client}/generate",
        json={
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": model_name,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )

    if response.status_code == 429:
        print("Rate limit hit - wait a moment and try again.")
        return None
    if response.status_code == 400:
        print(f"Bad request: {response.json().get('detail')}")
        return None

    response.raise_for_status()
    return response.json()

def build_prompt(question, choices):
    option_lines = [f"{letter}. {text}"
                    for letter, text in zip("ABCD", choices)]
    # write your own instruction below
    instruction = "Please answer with a single letter (A, B, C, or D) and nothing else."
    return f"{question}\n" + "\n".join(option_lines) + "\n\n" + instruction

def parse_single_letter(text):
    """Return the first standalone A/B/C/D found in text, or None."""
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None

def batch_questions(df, batch_size):
    """Split df into batches; each batch is (prompt_string, {qnum: correct_letter})."""
    prompts, answer_keys = [], []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        header = (
            f"Answer the following {len(batch)} multiple choice questions.\n"
            "For each question, answer with the question number followed by a "
            "letter (A, B, C, or D) and nothing else.\n\n"
        )
        body = ""
        key = {}
        for local_i, (_, row) in enumerate(batch.iterrows(), start=1):
            option_lines = [f"{letter}. {text}"
                            for letter, text in zip("ABCD", row["choices"])]
            body += f"{local_i}: {row['question']}\n" + "\n".join(option_lines) + "\n\n"
            key[local_i] = ANSWER_MAP[row["answer"]]
        prompts.append(header + body)
        answer_keys.append(key)
    return prompts, answer_keys

SYSTEM_PROMPT = """You are an answer key. You receive multiple choice questions and output only the answers.

RULES:
- Output one line per question.
- Each line is exactly: the question number, a colon, a space, and a single capital letter (A, B, C, or D).
- Output nothing else: no explanations, no restating the question, no extra words, no blank lines, no markdown.
- Answer every question. If unsure, still pick the single most likely letter.

EXAMPLE REPLY (for 3 questions):
1: A
2: C
3: B
"""

def parse_batched_answers(text):
    """Parse 'N: X' lines into {N: 'X'} keeping only valid A-D letters."""
    answers = {}
    for line in text.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*[:.]\s*([ABCD])", line.strip().upper())
        if m:
            answers[int(m.group(1))] = m.group(2)
    return answers

def score_batch(parsed, answer_key, verbose=False):
    """Return accuracy of one parsed batch against its answer key."""
    correct = 0
    for qnum, gold in answer_key.items():
        got = parsed.get(qnum)
        if got == gold:
            correct += 1
        if verbose:
            mark = "OK" if got == gold else " X"
            print(f"{mark}\tQ{qnum}: correct={gold}, model={got}")
    return correct / len(answer_key)

def evaluate_dataset_batched(df, model_name, client, batch_size=10, system_prompt=SYSTEM_PROMPT,
                             max_tokens=200, pause=0.5):
    from tqdm import tqdm
    import time
    prompts, answer_keys = batch_questions(df, batch_size=batch_size)
    n_correct, n_total = 0, 0
    for prompt, key in tqdm(list(zip(prompts, answer_keys)), desc="Batches"):
        res = query_model(prompt, model_name, max_tokens=max_tokens,
                          client=client, system_prompt=system_prompt)
        time.sleep(pause)  # be gentle on the shared rate limit
        if res is None:
            continue
        parsed = parse_batched_answers(res["answer"])
        for qnum, gold in key.items():
            n_total += 1
            if parsed.get(qnum) == gold:
                n_correct += 1
    return n_correct / n_total if n_total else 0.0

def parse_batched_answers_with_token_prob(batch_result):
    parsed_answers = {}
    for item in batch_result["logprobs_content"]:
        token = item['token'].strip()
        logprob = item['logprob']
        if re.match(r"^\d+$", token):
            current_question = int(token)
        elif re.match(r"^[ABCD]$", token):
            prob = np.exp(logprob)
            parsed_answers[current_question] = (token, prob)
    return parsed_answers

def collect_confidences_batched(df, model_name, client, batch_size=10, system_prompt=SYSTEM_PROMPT,
                                max_tokens=200, pause=0.5):
    from tqdm import tqdm
    import time
    prompts, answer_keys = batch_questions(df, batch_size=batch_size)
    rows = []
    for prompt, key in tqdm(list(zip(prompts, answer_keys)), desc="Batches"):
        res = query_model(prompt, model_name, max_tokens=max_tokens,
                          client=client, system_prompt=system_prompt)
        time.sleep(pause)
        if res is None:
            continue
        parsed = parse_batched_answers_with_token_prob(res)
        for qnum, gold in key.items():
            if qnum in parsed:
                letter, conf = parsed[qnum]
                rows.append({
                    "correct_answer": gold,
                    "model_answer": letter,
                    "is_correct": letter == gold,
                    "confidence": min(conf, 1.0),
                })
    return pd.DataFrame(rows)


def collect_confidences(df, model_name, client, max_questions=None, pause=0.4):
    """One call per question. Returns a DataFrame with the model's answer,
    whether it was correct, and the probability it assigned to its answer letter."""
    from tqdm import tqdm 
    import time
    if max_questions is not None:
        df = df.head(max_questions)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Questions"):
        gold = ANSWER_MAP[row["answer"]]
        prompt = build_prompt(row["question"], row["choices"])
        res = query_model(prompt, model_name, max_tokens=5, client=client)
        time.sleep(pause)
        if res is None:
            continue

        letter = parse_single_letter(res["answer"])
        if letter is None:
            continue

        # Confidence = probability of the answer-letter token.
        # token_probs is keyed by the raw token text, which may include a
        # leading space (e.g. " A"), so check a few variants.
        token_probs = res["token_probs"]
        conf = None
        for cand in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
            if cand in token_probs:
                conf = token_probs[cand]
                break
        if conf is None and token_probs:
            # fall back to the most probable generated token
            conf = max(token_probs.values())
        if conf is None:
            continue

        rows.append({
            "correct_answer": gold,
            "model_answer": letter,
            "is_correct": letter == gold,
            "confidence": min(conf, 1.0),
        })
    return pd.DataFrame(rows)

def compute_calibration(results_df, n_bins=10):
    """Bin by confidence and compute per-bin accuracy, confidence, and ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    results_df = results_df.copy()
    results_df["bin"] = pd.cut(results_df["confidence"], bins=bin_edges,
                               labels=False, include_lowest=True)

    cal = results_df.groupby("bin").agg(
        avg_confidence=("confidence", "mean"),
        avg_accuracy=("is_correct", "mean"),
        std_accuracy=("is_correct", "std"),
        count=("is_correct", "count"),
    ).reset_index()

    cal["accuracy_se"] = cal["std_accuracy"] / np.sqrt(cal["count"])
    cal = cal.fillna(0)
    cal["bin_center"] = cal["bin"].apply(
        lambda b: (bin_edges[int(b)] + bin_edges[int(b) + 1]) / 2.0
    )

    weight = cal["count"] / cal["count"].sum()
    ece = float((weight * (cal["avg_accuracy"] - cal["avg_confidence"]).abs()).sum())
    return cal, ece

def plot_calibration(cal, ece, title):
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(cal["bin_center"], cal["avg_accuracy"],
               color="C0", s=90, zorder=3, label="Observed accuracy")
    ax.errorbar(cal["bin_center"], cal["avg_accuracy"], yerr=cal["accuracy_se"],
                fmt="none", ecolor="gray", alpha=0.7, capsize=4, zorder=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    # annotate sample count per bin
    for _, r in cal.iterrows():
        if r["count"] > 0:
            ax.text(r["bin_center"], 0.02, f"N={int(r['count'])}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Predicted confidence (bin center)")
    ax.set_ylabel("Observed accuracy in bin")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=12, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.legend(bbox_to_anchor=(1.05, 0.9), loc="upper left")
    plt.show()

def load_subject(dataset_name, split="test"):
    BASE_URL = "hf://datasets/cais/mmlu/"
    path = f"{dataset_name}/{split}-00000-of-00001.parquet"
    return pd.read_parquet(BASE_URL + path)

def calibration_for(dataset_name, model_name, client, n_bins=10):
    """Full pipeline for one (subject, model): load -> query -> plot."""
    df = load_subject(dataset_name)
    res_df = collect_confidences_batched(df, model_name, client=client)
    if len(res_df) == 0:
        print(f"No results for {dataset_name} / {model_name}.")
        return None
    cal, ece = compute_calibration(res_df, n_bins=n_bins)
    short = model_name.split("/")[-1]
    plot_calibration(cal, ece,
                     f"Calibration - {short}\n{dataset_name} "
                     f"(N={len(res_df)}, acc={res_df['is_correct'].mean():.0%})")
    return {"dataset": dataset_name, "model": model_name,
            "n": len(res_df), "accuracy": res_df["is_correct"].mean(), "ece": ece}
