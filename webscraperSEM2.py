import praw
import os
import sys
import re
import time
import datetime as dt
from typing import Iterable, Dict, Set, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from key import API_KEY  # keep your existing key setup

def removeunknownCHAR(text: str) -> str:
    characters = re.compile(
        "["                     # a few common “noise” ranges
        "\U0001F600-\U0001F64F" # emojis
        "\U0001F300-\U0001F5FF" # symbols & pictographs
        "\U0001F680-\U0001F6FF" # transport & map symbols
        "\U0001F1E0-\U0001F1FF" # flags
        "\U00002700-\U000027BF" # dingbats
        "\U0001F900-\U0001F9FF" # supplemental symbols
        "\U00002600-\U000026FF" # miscellaneous symbols
        "\U00002500-\U00002BEF" # CJK (optional—remove if you want to keep CJK)
        "]+", flags=re.UNICODE
    )
    return characters.sub(r'', text or "")

def year_from_utc(ts: float) -> int:
    return dt.datetime.utcfromtimestamp(ts).year

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_post_by_year(base_dir: str, year: int, sub: str, post) -> None:
    ensure_dir(base_dir)
    fname = os.path.join(base_dir, f"collected_reddit_posts_{year}.txt")
    title = removeunknownCHAR(post.title)
    selftext = removeunknownCHAR(getattr(post, "selftext", "") or "")
    # Trim body a bit to keep files reasonable; tweak as needed
    body_preview = selftext[:2000]

    created_iso = dt.datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = (
        f"[{sub}] {title}\n"
        f"Author: u/{getattr(post.author, 'name', '[deleted]')}\n"
        f"Score: {post.score} | Comments: {post.num_comments}\n"
        f"Created (UTC): {created_iso}\n"
        f"URL: https://www.reddit.com{post.permalink}\n"
        f"ID: {post.id}\n"
        f"{body_preview}\n"
        f"--\n"
    )
    with open(fname, "a", encoding="utf-8") as f:
        f.write(block)

def looks_relevant(post, keywords_lower: List[str]) -> bool:
    # Extra guard: ensure at least one keyword appears in title or selftext.
    title = (post.title or "").lower()
    body  = (getattr(post, "selftext", "") or "").lower()
    return any(kw in title or kw in body for kw in keywords_lower)

def collect_by_year(
    client_id: str,
    client_secret: str,
    user_agent: str,
    subreddits: Iterable[str],
    keywords: Iterable[str],
    year_start: int = 2015,
    year_end: int = 2025,
    out_dir: str = "out",
    per_query_limit: int = 1000,
    sleep_between_calls: float = 1.5
):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    keywords = list(keywords)
    subreddits = list(subreddits)
    keywords_lower = [k.lower() for k in keywords]

    # Keep a global set of seen IDs to avoid duplicates across queries/keywords
    seen_ids: Set[str] = set()

    # Build individual keyword queries to maximize coverage (dedupe afterwards)
    # Use quotes around multi-word phrases and join with OR when we do combined runs.
    def quote_kw(kw: str) -> str:
        kw = kw.strip()
        return f"\"{kw}\"" if " " in kw else kw

    # Optional: a combined query pass (broad net) + per-keyword passes (depth)
    combined_query = " OR ".join(quote_kw(k) for k in keywords)

    # Loop across subreddits and run two passes: combined, then per-keyword
    for sub in subreddits:
        print(f"\n=== Subreddit: r/{sub} ===")
        # Pass 1: Combined query by relevance
        try:
            print(f"Search (combined): {combined_query}")
            for idx, post in enumerate(
                reddit.subreddit(sub).search(
                    combined_query,
                    sort="relevance",
                    time_filter="all",
                    limit=per_query_limit,
                    syntax="lucene",  # better OR behavior
                ),
                start=1
            ):
                if post.id in seen_ids:
                    continue
                y = year_from_utc(post.created_utc)
                if year_start <= y <= year_end and looks_relevant(post, keywords_lower):
                    write_post_by_year(out_dir, y, sub, post)
                    seen_ids.add(post.id)

                if idx % 100 == 0:
                    print(f"Processed {idx} posts (combined) in r/{sub}…")
            time.sleep(sleep_between_calls)
        except Exception as e:
            print(f"[WARN] Combined search failed in r/{sub}: {e}")

        # Pass 2: per-keyword (helps bypass search caps / recall more)
        for kw in keywords:
            q = quote_kw(kw)
            try:
                print(f"Search (keyword): {q}")
                for idx, post in enumerate(
                    reddit.subreddit(sub).search(
                        q,
                        sort="relevance",
                        time_filter="all",
                        limit=per_query_limit,
                        syntax="lucene",
                    ),
                    start=1
                ):
                    if post.id in seen_ids:
                        continue
                    y = year_from_utc(post.created_utc)
                    if year_start <= y <= year_end and looks_relevant(post, keywords_lower):
                        write_post_by_year(out_dir, y, sub, post)
                        seen_ids.add(post.id)

                    if idx % 100 == 0:
                        print(f"Processed {idx} posts for kw={q} in r/{sub}…")
                time.sleep(sleep_between_calls)
            except Exception as e:
                print(f"[WARN] Keyword search failed for {q} in r/{sub}: {e}")

    print("\nDone! Files written per-year in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    # —— Configure your inputs here ——
    KEYWORDS = [
        "Student evaluation of teaching", "course evaluation",
        "student evals", "SETs", "SET", "student evaluations"
    ]
    SUBREDDITS = ["Professors"]  # add "AskAcademia", "AskProfessors", etc.

    collect_by_year(
        client_id="CzN425wWb3G61s3vGyy3mw",
        client_secret=API_KEY,
        user_agent="Education Query 1.0 A /u/eloiseava",
        subreddits=SUBREDDITS,
        keywords=KEYWORDS,
        year_start=2015,
        year_end=2025,
        out_dir="out",                 # each year gets its own txt in this dir
        per_query_limit=1000,          # bump down if you hit limits
        sleep_between_calls=1.75       # gentle on rate limits
    )
