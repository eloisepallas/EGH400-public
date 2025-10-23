# pip install openai python-dotenv
import os, re, json, time, csv, glob
from statistics import mean
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Iterable, Optional
from openai import OpenAI

# ---------- Config ----------
YEAR_START = 2023
YEAR_END   = 2025
IN_DIR     = "out"        # where your yearly txts live
OUT_DIR    = "analysis"   # where we'll write CSVs
PER_POST_SLEEP = 0.0      # increase if you want to throttle requests

# Model & prompt config
MODEL = "gpt-5-mini"

SYSTEM = """Role: You are an expert in sentiment and emotion analysis with advanced emotional intelligence.
Purpose: This analysis will be used for a university faculty feedback study to identify patterns of emotional tone in student evaluations for teaching improvement."""

# Fixed emotion list (28)
EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

# ---------- Init ----------
load_dotenv("key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Model call (parameterized) ----------
def call_model_get_scores(text: str) -> Dict:
    """
    Calls the model for a given text and returns JSON strictly matching the schema.
    """
    user_prompt = f"""
Analyse the provided text and assign each emotion a score between 0.00 and 1.00 (two decimals).
Return JSON only, matching the schema: one named score per emotion, a list of top3 emotions, and evidence spans.

Text to analyse:
\"\"\"{text}\"\"\""""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "emotion_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "object",
                            "properties": {emo: {"type": "number"} for emo in EMOTIONS},
                            "required": EMOTIONS
                        },
                        "top3": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3
                        },
                        "evidence": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": ["scores", "top3", "evidence"]
                }
            }
        }
    )

    data = json.loads(resp.choices[0].message.content)

    # Validate schema
    scores = data.get("scores", {})
    if len(scores) != len(EMOTIONS):
        raise ValueError(f"Expected {len(EMOTIONS)} emotions, got {len(scores)}")
    return data


# ---------- Stabilization (unchanged logic, now accepts text) ----------
def running_avg_update(avg: Dict[str,float], new: Dict[str,float], n: int) -> Dict[str,float]:
    if n == 1:
        return dict(new)
    return {emo: a + (new[emo] - a)/n for emo, a in avg.items()}

def top3_from_scores(scores: Dict[str,float]) -> List[str]:
    return sorted(scores.keys(), key=lambda e: scores[e], reverse=True)[:3]

def stabilize_scores_for_text(
    text: str,
    eps_top3: float = 0.05,
    max_iters: int = 25,
    sleep_s: float = 0.0
) -> Dict:
    running_avg = None
    prev_avg = None
    n = 0
    history = []
    reason = "Hit max_iters"

    while n < max_iters:
        data = call_model_get_scores(text)
        scores = {emo: float(f"{float(data['scores'][emo]):.2f}") for emo in EMOTIONS}
        history.append(scores)
        n += 1

        running_avg = running_avg_update(running_avg or scores, scores, n)

        if prev_avg is not None:
            idx = top3_from_scores(running_avg)
            max_delta = max(abs(running_avg[e] - prev_avg[e]) for e in idx)
            if max_delta <= eps_top3:
                reason = f"Converged: top-3 running-average change ≤ {eps_top3:.2f}"
                break

        prev_avg = dict(running_avg)
        if sleep_s > 0:
            time.sleep(sleep_s)

    top3_emotions = top3_from_scores(running_avg)
    final_scores = {emo: round(running_avg[emo], 2) for emo in EMOTIONS}
    return {
        "iterations": n,
        "stop_reason": reason,
        "top3": top3_emotions,
        "scores": final_scores,
        "history_len": len(history),
        "history": history
    }


# ---------- I/O helpers ----------
POST_SPLIT_RE = re.compile(r"\n--\s*\n", re.MULTILINE)

def iter_posts_from_year_file(path: str) -> Iterable[Dict]:
    """
    Yields dicts: {index, raw_block, text, id, url, created, title}
    The collector wrote blocks like:
        [sub] TITLE
        Author: ...
        Score: ...
        Created (UTC): ...
        URL: ...
        ID: ...
        BODY...
        --
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return

    blocks = POST_SPLIT_RE.split(content)
    for i, block in enumerate(blocks, start=1):
        b = block.strip()
        if not b:
            continue
        # Extract a few metadata fields if present
        title_match = re.match(r"\[.*?\]\s*(.*)", b.splitlines()[0]) if b.splitlines() else None
        title = title_match.group(1).strip() if title_match else ""

        def find_line(prefix: str) -> Optional[str]:
            for line in b.splitlines():
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
            return None

        post_id = find_line("ID: ")
        url     = find_line("URL: ")
        created = find_line("Created (UTC): ")

        # For the model, we can pass the whole block (title + body + some meta)
        yield {
            "index": i,
            "raw_block": b,
            "text": b,
            "id": post_id or "",
            "url": url or "",
            "created": created or "",
            "title": title or ""
        }


def write_per_post_csv(year: int, rows: List[Dict[str, str]]) -> str:
    """
    Writes a wide CSV with one row per post: meta + all emotion columns.
    """
    out_path = os.path.join(OUT_DIR, f"per_post_{year}.csv")
    fieldnames = [
        "year","post_index","post_id","url","created","title","iterations","stop_reason","top3"
    ] + EMOTIONS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path


def write_year_summary_csv(year: int, per_post_rows: List[Dict[str, str]]) -> str:
    """
    Writes one row with mean score per emotion and count of posts.
    """
    out_path = os.path.join(OUT_DIR, f"summary_{year}.csv")
    n = len(per_post_rows)
    means = {}
    if n > 0:
        for emo in EMOTIONS:
            means[emo] = mean(float(r[emo]) for r in per_post_rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year","num_posts"] + EMOTIONS)
        if n > 0:
            w.writerow([year, n] + [f"{means[emo]:.4f}" for emo in EMOTIONS])
        else:
            w.writerow([year, 0] + ["0.0000"]*len(EMOTIONS))
    return out_path


def append_master_summary(year: int, per_post_rows: List[Dict[str, str]], master_path: str):
    """
    Appends (or creates) a master CSV with one row per year (num_posts + mean per emotion).
    """
    n = len(per_post_rows)
    if n > 0:
        means = {emo: mean(float(r[emo]) for r in per_post_rows) for emo in EMOTIONS}
    else:
        means = {emo: 0.0 for emo in EMOTIONS}

    file_exists = os.path.exists(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["year","num_posts"] + EMOTIONS)
        w.writerow([year, n] + [f"{means[emo]:.4f}" for emo in EMOTIONS])


# ---------- Main driver ----------
def analyze_all_years(
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    eps_top3: float = 0.05,
    max_iters: int = 25
):
    master_path = os.path.join(OUT_DIR, "master_summary.csv")
    if os.path.exists(master_path):
        os.remove(master_path)

    for year in range(year_start, year_end + 1):
        in_path = os.path.join(IN_DIR, f"collected_reddit_posts_{year}.txt")
        if not os.path.exists(in_path):
            print(f"[{year}] No input file found at {in_path}; skipping.")
            append_master_summary(year, [], master_path)
            continue

        print(f"\n=== Year {year} ===")
        per_post_rows: List[Dict[str, str]] = []

        for post in iter_posts_from_year_file(in_path):
            idx = post["index"]
            text = post["text"]  # could also choose just title + body; we pass the block
            try:
                result = stabilize_scores_for_text(
                    text=text,
                    eps_top3=eps_top3,
                    max_iters=max_iters,
                    sleep_s=0.0
                )
            except Exception as e:
                # If one post fails, record an error row and continue
                print(f"[{year} #{idx}] ERROR: {e}")
                row = {
                    "year": str(year),
                    "post_index": str(idx),
                    "post_id": post["id"],
                    "url": post["url"],
                    "created": post["created"],
                    "title": post["title"][:200],
                    "iterations": "0",
                    "stop_reason": f"ERROR: {e}",
                    "top3": ""
                }
                for emo in EMOTIONS:
                    row[emo] = "0.00"
                per_post_rows.append(row)
                continue

            # Build CSV row
            row = {
                "year": str(year),
                "post_index": str(idx),
                "post_id": post["id"],
                "url": post["url"],
                "created": post["created"],
                "title": post["title"][:200],
                "iterations": str(result["iterations"]),
                "stop_reason": result["stop_reason"],
                "top3": ";".join(result["top3"])
            }
            for emo in EMOTIONS:
                row[emo] = f"{result['scores'][emo]:.2f}"
            per_post_rows.append(row)

            if PER_POST_SLEEP > 0:
                time.sleep(PER_POST_SLEEP)

            # Optional: lightweight progress
            if idx % 10 == 0:
                print(f"[{year}] processed {idx} posts…")

        # Write per-year outputs
        per_post_csv = write_per_post_csv(year, per_post_rows)
        year_summary_csv = write_year_summary_csv(year, per_post_rows)
        append_master_summary(year, per_post_rows, master_path)

        print(f"[{year}] Wrote: {per_post_csv}")
        print(f"[{year}] Wrote: {year_summary_csv}")

    print("\nDone.")
    print(f"Master summary -> {os.path.join(OUT_DIR, 'master_summary.csv')}")


if __name__ == "__main__":
    # Tweak eps_top3 / max_iters as you like
    analyze_all_years(eps_top3=0.05, max_iters=25)
