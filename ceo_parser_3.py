import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from langdetect import detect, DetectorFactory, LangDetectException
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk import sent_tokenize
import re

DetectorFactory.seed = 0

# Params define company name and main-page url necessary for the functioning of this script.
params = [company_url, company_name]  #Both str


def crawl_site(base_url, delay=0.5, max_pages=200):
    visited = set()
    to_visit = [base_url]
    domain = urlparse(base_url).netloc
    pages = {}

    def is_valid_link(link):
        if not link:
            return False
        link = link.strip()
        invalid_prefixes = ("mailto:", "javascript:", "tel:", "#")
        if link.startswith(invalid_prefixes):
            return False
        invalid_exts = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".rar", ".exe")
        if any(link.lower().endswith(ext) for ext in invalid_exts):
            return False
        return True

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
            pages[url] = html
            visited.add(url)

            soup = BeautifulSoup(html, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag['href']
                if not is_valid_link(href):
                    continue
                abs_url = urljoin(url, href)
                parsed = urlparse(abs_url)
                if parsed.netloc == domain and abs_url not in visited and abs_url not in to_visit:
                    to_visit.append(abs_url)

            time.sleep(delay)

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            visited.add(url)

    return pages


def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    content_tags = ['p', 'h1', 'h2', 'h3', 'li']
    text_blocks = []
    for tag in soup.find_all(content_tags):
        text = tag.get_text(separator=' ', strip=True)
        if text:
            text_blocks.append(text)
    return text_blocks


def detect_language(text):
    try:
        if len(text.strip()) < 20:
            return "unknown"
        lang = detect(text)
        if lang in ["en", "es"]:
            return lang
        else:
            return "unknown"
    except LangDetectException:
        return "unknown"


_nlp_models = {}


def load_spacy_model(lang):
    if lang == "en":
        model_name = "en_core_web_sm"
    elif lang == "es":
        model_name = "es_core_news_sm"
    else:
        return None

    if lang not in _nlp_models:
        try:
            _nlp_models[lang] = spacy.load(model_name)
        except Exception as e:
            print(f"Could not load spaCy model for {lang}: {e}")
            return None

    return _nlp_models[lang]


def ner_extract(sentence, lang):
    nlp = load_spacy_model(lang)
    if nlp is None:
        return {"persons": np.nan, "orgs": np.nan}
    doc = nlp(sentence)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PER" or ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return {
        "persons": persons if persons else np.nan,
        "orgs": orgs if orgs else np.nan
    }


def filter_by_keywords(sentence, keyword_groups=None):
    sentence_lower = sentence.lower()
    if keyword_groups is None:
        keyword_groups = [
            ["founder", "fundador", "fundadora"],  # rank 1
            ["ceo", "chief executive", "director general"],  # rank 2
            ["president", "presidente"],  # rank 3
            ["co-founder", "cofundador", "cofundadora"],  # rank 4
        ]
    for rank, group in enumerate(keyword_groups, start=1):
        for keyword in group:
            if keyword in sentence_lower:
                return {
                    "matched": True,
                    "keyword": keyword,
                    "rank": rank
                }
    return {"matched": False, "keyword": None, "rank": None}


def rank_candidates_by_similarity(candidates, company_name):
    if not candidates:
        return None

    # Filter out candidates missing org or person or with nan values
    valid = [
        c for c in candidates
        if c.get("org") and c.get("person")
           and isinstance(c.get("org"), str)
           and isinstance(c.get("person"), str)
    ]

    if not valid:
        return None

    org_names = [c["org"] for c in valid]
    texts = org_names + [company_name]

    vectorizer = TfidfVectorizer().fit(texts)
    org_vectors = vectorizer.transform(org_names)
    input_vector = vectorizer.transform([company_name])

    sims = cosine_similarity(org_vectors, input_vector).flatten()

    for i, c in enumerate(valid):
        c["similarity"] = sims[i]

    best = sorted(valid, key=lambda x: (x["role_rank"], -x["similarity"]))[0]

    return {
        "name": best["person"],
        "role": best["keyword"],
        "sentence": best["sentence"]
    }


def extract_emails(text):
    matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if matches:
        print("Extracted emails:", matches)
        return matches
    print("No email found.")
    return []


def rank_emails_by_similarity(emails, company_name):
    company_name_lower = company_name.lower()
    for email in emails:
        email_lower = email.lower()
        if company_name_lower in email_lower:
            return email
    return emails[0] if emails else None  # fallback to first email if none matc


def run_pipeline_and_output(root_url, company_name, output_csv="ceo_result.csv"):
    print("Crawling site...")
    pages = crawl_site(root_url)

    print(f"Extracting and joining text from {len(pages)} pages...")
    all_text_blocks = []
    for html in pages.values():
        all_text_blocks.extend(extract_visible_text(html))

    full_text = " ".join(all_text_blocks)
    print(f"Total combined text length: {len(full_text)} characters")

    # Extract emails from combined_text (no new variable names)
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    all_emails = extract_emails(full_text)

    top_email = rank_emails_by_similarity(all_emails, company_name)
    print("Top company email candidate:", top_email)

    print("Splitting into sentences...")
    sentences = sent_tokenize(full_text)

    all_candidates = []
    for sentence in sentences:
        lang = detect_language(sentence)
        if lang not in ['en', 'es']:
            continue

        ner_result = ner_extract(sentence, lang)
        kw_result = filter_by_keywords(sentence)

        if not kw_result["matched"]:
            continue

        persons = ner_result["persons"]
        orgs = ner_result["orgs"]

        if isinstance(persons, list):
            for person in persons:
                org = orgs[0] if isinstance(orgs, list) and orgs else np.nan
                all_candidates.append({
                    "sentence": sentence,
                    "person": " ".join(person.split()[:2]),
                    "org": org,
                    "role_rank": kw_result["rank"],
                    "keyword": kw_result["keyword"]
                })

    top = rank_candidates_by_similarity(all_candidates, company_name)
    print("Top founder candidates:", top)

    df = pd.DataFrame([top] if top else [], columns=["name", "role", "sentence"])

    # Add email column with existing variable names
    df['email'] = top_email if top_email else None

    df.to_csv(output_csv, index=False)

    if top:
        print("✅ Likely match found:")
        print(df)
    else:
        print("⚠️ No match found.")


if __name__ == "__main__":
    run_pipeline_and_output(params[0], params[1])
