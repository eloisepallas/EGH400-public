# build_corpus.py
import os, re, pandas as pd
from pathlib import Path

YEAR_START, YEAR_END = 2015, 2025
IN_DIR   = "out"        # where collected_reddit_posts_YYYY.txt live
AN_DIR   = "analysis"   # where per_post_YYYY.csv live (from your run)
OUT_PATH = os.path.join(AN_DIR, "corpus.csv")

POST_SPLIT_RE = re.compile(r"\n--\s*\n", re.MULTILINE)

def parse_year_file(year_path: str, year: int) -> pd.DataFrame:
    with open(year_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return pd.DataFrame()

    blocks = POST_SPLIT_RE.split(content)
    rows = []
    for i, block in enumerate(blocks, start=1):
        b = block.strip()
        if not b:
            continue
        lines = b.splitlines()
        title = ""
        if lines:
            m = re.match(r"\[.*?\]\s*(.*)", lines[0])
            title = m.group(1).strip() if m else lines[0].strip()

        def find_line(prefix: str):
            for ln in lines:
                if ln.startswith(prefix):
                    return ln[len(prefix):].strip()
            return ""

        rows.append({
            "year": year,
            "post_index": i,
            "post_id": find_line("ID: "),
            "url": find_line("URL: "),
            "created": find_line("Created (UTC): "),
            "title": title,
            "text": b,  # keep full block for now; you can trim later
        })
    return pd.DataFrame(rows)

def main():
    Path(AN_DIR).mkdir(exist_ok=True)
    dfs = []
    for y in range(YEAR_START, YEAR_END+1):
        p = os.path.join(IN_DIR, f"collected_reddit_posts_{y}.txt")
        if os.path.exists(p):
            df = parse_year_file(p, y)
            dfs.append(df)
    corpus = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Merge in emotions from per_post_YYYY.csv (if present)
    emo_cols = None
    if not corpus.empty:
        merged = []
        for y, g in corpus.groupby("year"):
            per_post_path = os.path.join(AN_DIR, f"per_post_{y}.csv")
            if os.path.exists(per_post_path):
                per = pd.read_csv(per_post_path)
                if emo_cols is None:
                    emo_cols = [c for c in per.columns
                                if c not in ("year","post_index","post_id","url","created","title",
                                             "iterations","stop_reason","top3")]
                # merge on (year, post_index) which your analyzer wrote
                mg = pd.merge(g, per[["year","post_index","title"]+emo_cols], on=["year","post_index","title"], how="left")
            else:
                mg = g
            merged.append(mg)
        corpus = pd.concat(merged, ignore_index=True)

    corpus.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(corpus)} rows")

if __name__ == "__main__":
    main()
