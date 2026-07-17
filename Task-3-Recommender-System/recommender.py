"""
==========================================================
 PROJECT 3 : AI RECOMMENDATION LOGIC
 DecodeLabs - Industrial Training Kit
==========================================================


"""

import csv
import math
import os
from collections import Counter

# Folder where THIS script lives (so it works no matter which
# folder you run "python recommender.py" from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "raw_skills.csv")


# ----------------------------------------------------------
# STEP 0: LOAD DATA
# ----------------------------------------------------------
def load_dataset(path=DEFAULT_CSV_PATH):
    """Reads raw_skills.csv -> list of (job_role, [skill_tags])"""
    roles = []
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'.\n"
            f"Make sure 'raw_skills.csv' is in the SAME folder as recommender.py:\n"
            f"  {SCRIPT_DIR}"
        )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            tags = [t.strip().lower() for t in row["skills"].split(",")]
            roles.append((role, tags))
    return roles


# ----------------------------------------------------------
# STEP 1: BUILD VOCABULARY  (shared vector space)
# ----------------------------------------------------------
def build_vocabulary(documents):
    """
    documents: list of tag-lists (each item's skills, plus the
    user profile at the end). Returns a sorted list of every
    unique skill term across all documents -- this is the
    'shared vocabulary space' from the slides.
    """
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    return sorted(vocab)


# ----------------------------------------------------------
# STEP 2: TF-IDF WEIGHTING
# ----------------------------------------------------------
def compute_tf(doc_tags, vocab):
    """Term Frequency: count(term) / total_terms_in_doc"""
    counts = Counter(doc_tags)
    total = len(doc_tags) if doc_tags else 1
    return {term: counts.get(term, 0) / total for term in vocab}


def compute_idf(documents, vocab):
    """
    Inverse Document Frequency:
        idf(t) = log( N / (1 + docs_containing_t) )
    The '+1' avoids division-by-zero for terms unique to one doc.
    """
    n_docs = len(documents)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for doc in documents if term in doc)
        idf[term] = math.log(n_docs / (1 + docs_with_term)) + 1  # smoothed
    return idf


def tfidf_vector(doc_tags, vocab, idf):
    """Combines TF and IDF into a single weighted vector (list of floats)."""
    tf = compute_tf(doc_tags, vocab)
    return [tf[term] * idf[term] for term in vocab]


# ----------------------------------------------------------
# STEP 3: COSINE SIMILARITY  (the Similarity Engine)
# ----------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    """
        cos(theta) = (A . B) / (||A|| * ||B||)
    Score of 1  -> perfectly aligned (great match)
    Score of 0  -> orthogonal (no shared interests)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold-Start case: zero vector -> zero score
    return dot_product / (magnitude_a * magnitude_b)


# ----------------------------------------------------------
# STEP 4: THE FULL 4-STEP PIPELINE
# ----------------------------------------------------------
def recommend(user_skills, dataset_path=DEFAULT_CSV_PATH, top_n=3):
    """
    user_skills: list of strings, e.g. ["Python", "Cloud", "Automation"]
    Returns Top-N (job_role, score) tuples, sorted descending.
    """
    # ---- 1. INGESTION ----
    if len(user_skills) < 3:
        raise ValueError("Please provide at least 3 skills (data density requirement).")
    user_tags = [s.strip().lower() for s in user_skills]

    roles = load_dataset(dataset_path)

    # Cold-Start fallback: if none of the user's skills exist anywhere
    # in the dataset vocabulary, fall back to "Trending" (most common skills)
    all_known_tags = set(tag for _, tags in roles for tag in tags)
    if not any(tag in all_known_tags for tag in user_tags):
        return trending_fallback(roles, top_n)

    documents = [tags for _, tags in roles] + [user_tags]
    vocab = build_vocabulary(documents)
    idf = compute_idf(documents, vocab)

    user_vector = tfidf_vector(user_tags, vocab, idf)

    # ---- 2. SCORING ----
    scored = []
    for role, tags in roles:
        role_vector = tfidf_vector(tags, vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scored.append((role, round(score, 4)))

    # ---- 3. SORTING ----
    scored.sort(key=lambda x: x[1], reverse=True)

    # ---- 4. FILTERING (Top-N) ----
    return scored[:top_n]


def trending_fallback(roles, top_n=3):
    """User Cold-Start bypass: return the most 'generic'/popular roles
    (here: the roles with the most listed skills, as a simple popularity proxy)."""
    popularity = [(role, len(tags)) for role, tags in roles]
    popularity.sort(key=lambda x: x[1], reverse=True)
    return [(role, 0.0) for role, _ in popularity[:top_n]]


# ----------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------
def main():
    print("=" * 55)
    print(" TECH STACK RECOMMENDER  |  Project 3 - DecodeLabs")
    print("=" * 55)
    print("Enter at least 3 skills, separated by commas.")
    print("Example: Python, Cloud, Automation\n")

    raw_input_str = input("Your skills: ")
    user_skills = [s.strip() for s in raw_input_str.split(",") if s.strip()]

    try:
        results = recommend(user_skills, top_n=3)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    print("\nTop 3 Recommended Career Paths:")
    print("-" * 40)
    for rank, (role, score) in enumerate(results, start=1):
        match_pct = round(score * 100, 1)
        print(f"{rank}. {role:<25} match: {match_pct}%")
    print("-" * 40)


if __name__ == "__main__":
    main()