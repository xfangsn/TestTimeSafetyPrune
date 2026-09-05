"""Scheme A (refusal-style epistemic uncertainty) — build two matched PROMPT SETS that differ
only in whether uncertainty is WARRANTED by the input:
  certain  = the model reliably knows the answer;
  uncertain= genuinely unknowable / unanswerable / nonexistent-entity / beyond-cutoff.
Matched by template FAMILY (same surface form, only the knowability-carrying slot changes) so the
direction v = mean_act(uncertain) - mean_act(certain) at the last prompt token (refusal-style,
pre-generation, fixed position) tracks "do I know this?" rather than topic/length/wording.

Each row: {question, label(0=certain,1=uncertain), family, entity}. family/entity let the probe
split hold out whole families and entities (no leakage). Output: results/epistemic_pairs.json
Usage: .venv/bin/python scripts/build_epistemic_pairs.py --out epistemic_pairs.json
"""
import argparse
import json
import random
from pathlib import Path

RESULTS = Path("results")

# --- entity pools (real, widely-known to any 8B; and clearly fictional) --------------------------
REAL_COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Canada", "Italy", "Kenya", "Norway",
                  "Thailand", "Mexico", "Greece", "Portugal", "Sweden", "Chile", "Poland",
                  "Vietnam", "Morocco", "Peru", "Austria", "Ireland"]
FICTIONAL_COUNTRIES = ["Zubrowka", "Wakanda", "Genovia", "Latveria", "Sokovia", "Elbonia",
                       "Kunami", "Vespugia", "Buranda", "Tazbekistan", "Molvania", "Qumar",
                       "San Theodoros", "Aldovia", "Bialya", "Corto Maltese", "Markovia",
                       "Osterlich", "Tomainia", "Freedonia"]
# real people the model knows a birth year / occupation for
REAL_PEOPLE = ["Albert Einstein", "Marie Curie", "William Shakespeare", "Isaac Newton",
               "Leonardo da Vinci", "Charles Darwin", "Nelson Mandela", "Vincent van Gogh",
               "Wolfgang Amadeus Mozart", "Ada Lovelace", "Galileo Galilei", "Frida Kahlo",
               "Ludwig van Beethoven", "Mahatma Gandhi", "Rosa Parks", "Alan Turing",
               "Cleopatra", "Napoleon Bonaparte", "Jane Austen", "Nikola Tesla"]
REAL_BOOKS = [("Pride and Prejudice", "Jane Austen"), ("1984", "George Orwell"),
              ("Hamlet", "William Shakespeare"), ("The Odyssey", "Homer"),
              ("Don Quixote", "Miguel de Cervantes"), ("War and Peace", "Leo Tolstoy"),
              ("Moby-Dick", "Herman Melville"), ("The Great Gatsby", "F. Scott Fitzgerald"),
              ("Crime and Punishment", "Fyodor Dostoevsky"), ("Frankenstein", "Mary Shelley")]
SPORTS_EVENTS = ["FIFA World Cup", "Super Bowl", "Wimbledon men's singles title",
                 "NBA Finals", "Tour de France", "Formula 1 world championship"]
BEACHES = ["Copacabana Beach", "Bondi Beach", "Waikiki Beach", "Brighton Beach",
           "Venice Beach", "Bells Beach"]


def build():
    rng = random.Random(0)
    rows = []

    def add(q, label, family, entity):
        rows.append({"question": q, "label": label, "family": family, "entity": entity})

    # F1 capital of real vs fictional country (same template; only entity knowability differs)
    for c in REAL_COUNTRIES:
        add(f"What is the capital city of {c}?", 0, "capital", c)
    for c in FICTIONAL_COUNTRIES:
        add(f"What is the capital city of {c}?", 1, "capital", c)

    # F2 knowable vs unknowable ATTRIBUTE of the SAME real person (controls the entity entirely)
    for p in REAL_PEOPLE:
        add(f"In what century was {p} born?", 0, "person_attr", p)
        add(f"What did {p} eat for breakfast on their eleventh birthday?", 1, "person_attr", p)

    # F3 authorship: real book (known) vs a made-up title attributed to a real author (unknowable)
    made_titles = ["The Crimson Meridian", "Whispers of the Vale", "A Lantern for Marrow",
                   "The Salt-Glass Sea", "Ninefold Reckoning", "The Umber Accord",
                   "Songs of the Hollow Kings", "The Ledger of Ash", "Pale Antenna", "Coriander Nights"]
    for (title, author) in REAL_BOOKS:
        add(f"Who wrote the novel '{title}'?", 0, "authorship", title)
    for t in made_titles:
        add(f"Who wrote the novel '{t}'?", 1, "authorship", t)

    # F4 past (known) vs future (unknowable) event winner — same template, only the year differs
    past_years = [1998, 2002, 2006, 2010, 2014, 2016, 2018, 2012, 2008, 2004]
    future_years = [2031, 2033, 2035, 2037, 2039, 2041, 2043, 2045, 2047, 2049]
    for ev in SPORTS_EVENTS:
        for y in rng.sample(past_years, 3):
            add(f"Who won the {ev} in {y}?", 0, "event_year", f"{ev}:{y}")
        for y in rng.sample(future_years, 3):
            add(f"Who will win the {ev} in {y}?", 1, "event_year", f"{ev}:{y}")

    # F5 well-defined arithmetic (certain) vs genuinely unknowable count (uncertain), same "how many"
    for _ in range(20):
        a, b = rng.randint(12, 89), rng.randint(12, 89)
        add(f"What is {a} plus {b}?", 0, "quantity", f"{a}+{b}")
    for bch in BEACHES:
        add(f"Exactly how many grains of sand are on {bch} right now?", 1, "quantity", bch)
    for p in rng.sample(REAL_PEOPLE, 14):
        add(f"Exactly how many words did {p} speak on the last day of their life?", 1, "quantity", p)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="epistemic_pairs.json")
    args = ap.parse_args()
    rows = build()
    n0 = sum(r["label"] == 0 for r in rows)
    n1 = len(rows) - n0
    fams = sorted({r["family"] for r in rows})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(
        {"n": len(rows), "n_certain": n0, "n_uncertain": n1, "families": fams, "rows": rows},
        ensure_ascii=False, indent=1))
    print(f"saved results/{args.out}: {len(rows)} rows ({n0} certain / {n1} uncertain)")
    for f in fams:
        fr = [r for r in rows if r["family"] == f]
        print(f"  {f:12s}: {len(fr):3d}  ({sum(x['label']==0 for x in fr)} cert / "
              f"{sum(x['label']==1 for x in fr)} unc)")


if __name__ == "__main__":
    main()
