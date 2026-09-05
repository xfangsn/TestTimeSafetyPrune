"""Experiment B (codex's decisive test) — counterfactual EVIDENCE-SUFFICIENCY crossover that breaks the
knowability<->surface-anomaly confound baked into scheme A's templated pairs.

2x2 per item quadruple, identical question wording:
  entity  in {familiar (real country), nonce (invented country)}
  context in {contains the answer, omits the answer}
The queried attribute is deliberately ARBITRARY/obscure (a ceremonial mace name, an archive founding
year, an official gemstone, ...) so NEITHER a familiar nor a nonce entity can be answered from parametric
knowledge — answerability is determined PURELY by whether the passage states it, orthogonal to entity
novelty. Instruction forces context-only answering.

Appropriate behavior: context-contains -> give the stated value; context-omits -> say it is not stated,
REGARDLESS of entity familiarity. So "should I abstain?" tracks EVIDENCE, not novelty. A weight edit that
removes abstention following evidence (both omits cells) = epistemic; one that follows novelty (only nonce
cells) = anomaly/novelty detector.

Output: results/evidence_crossover.json  Usage: .venv/bin/python scripts/build_evidence_crossover.py
"""
import argparse
import json
import random
from pathlib import Path

RESULTS = Path("results")

FAMILIAR = ["France", "Japan", "Brazil", "Canada", "Italy", "Kenya", "Norway", "Thailand", "Mexico",
            "Greece", "Portugal", "Sweden", "Chile", "Poland", "Vietnam", "Morocco", "Peru", "Austria",
            "Ireland", "Spain", "Argentina", "Finland", "Turkey", "Colombia", "Hungary", "Denmark",
            "Nigeria", "Malaysia", "Ecuador", "Croatia"]
NONCE = ["Kesteria", "Brownistan", "Veltoria", "Drennmoor", "Pelagon", "Yavthar", "Corvanth", "Malgrave",
         "Threnovia", "Ostrembia", "Kelmaria", "Vantasq", "Brindolia", "Quthara", "Sarnovia", "Weltavia",
         "Fenwold", "Glonteria", "Marovek", "Tesalburg", "Nyxoria", "Palvernia", "Cralovia", "Umbraq",
         "Zelmoor", "Vondaria", "Grivenland", "Torvaskia", "Belmoria", "Cindervale"]

# each attribute: (question phrasing, sentence stating the value, a value generator)
ATTRS = [
    ("the name of {X}'s national ceremonial mace",
     "The national ceremonial mace of {X} is called the {V}.", lambda r: f"Mace of {r.choice(MWORD)}"),
    ("the founding year of the {X} National Archive",
     "The {X} National Archive was founded in {V}.", lambda r: str(r.randint(1801, 1979))),
    ("the official state gemstone of {X}",
     "The official state gemstone of {X} is {V}.", lambda r: r.choice(GEMS)),
    ("the name of {X}'s highest civilian honour",
     "The highest civilian honour of {X} is the {V}.", lambda r: f"Order of the {r.choice(MWORD)}"),
    ("the colour of the sash worn by {X}'s head of state",
     "The head of state of {X} wears a {V} ceremonial sash.", lambda r: r.choice(COLORS)),
    ("the name of the official residence of {X}'s head of state",
     "The official residence of {X}'s head of state is called {V}.", lambda r: f"{r.choice(MWORD)} House"),
]
MWORD = ["Silver Heron", "Amber Vale", "Iron Larch", "Copper Wren", "Pale Meridian", "Golden Thistle",
         "Slate Harrier", "Crimson Alder", "Verdant Cairn", "Onyx Falcon", "Willow Marches", "Ashen Quill"]
GEMS = ["a pale-blue aquamarine", "a green tourmaline", "a banded agate", "a violet spinel",
        "a smoky topaz", "a rose garnet", "a golden beryl", "a grey moonstone"]
COLORS = ["deep crimson", "forest green", "royal violet", "burnt amber", "slate blue", "ochre yellow"]
# neutral facts for the OMITS passage (never the queried attribute)
FILLER = ["It has a temperate climate across most of its territory.",
          "Its economy relies on a mix of agriculture and light manufacturing.",
          "It is divided into several administrative provinces.",
          "Its cuisine features a variety of regional breads and stews.",
          "Public transport in its largest cities is widely used.",
          "It maintains diplomatic missions in dozens of countries.",
          "Its terrain includes both coastal plains and interior highlands.",
          "Seasonal festivals are held in many of its towns."]


def passage(rng, X, kind, value_sentence):
    body = rng.sample(FILLER, 3)
    if kind == "contains":
        body.insert(rng.randint(0, 3), value_sentence)
    return f"{X} is a country. " + " ".join(body)


def build(n_per_attr=6, seed=0):
    rng = random.Random(seed)
    rows = []
    qid = 0
    for ai, (qtmpl, vtmpl, vgen) in enumerate(ATTRS):
        fam_pool = rng.sample(FAMILIAR, n_per_attr)
        non_pool = rng.sample(NONCE, n_per_attr)
        for k in range(n_per_attr):
            for entity_kind, X in (("familiar", fam_pool[k]), ("nonce", non_pool[k])):
                value = vgen(rng)
                vsent = vtmpl.format(X=X, V=value)
                q = "According to the passage, what is " + qtmpl.format(X=X) + "?"
                for ctx in ("contains", "omits"):
                    p = passage(rng, X, ctx, vsent)
                    instr = (f"{p}\n\nUsing ONLY the passage above, answer the question. If the passage "
                             f"does not state the answer, reply that it is not stated.\n\nQuestion: {q}")
                    rows.append({"qid": qid, "attr": ai, "entity_kind": entity_kind, "entity": X,
                                 "context": ctx, "answerable": ctx == "contains",
                                 "value": value if ctx == "contains" else None,
                                 "question": instr})
            qid += 1
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="evidence_crossover.json")
    ap.add_argument("--n-per-attr", type=int, default=6); args = ap.parse_args()
    rows = build(args.n_per_attr)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False, indent=1))
    from collections import Counter
    c = Counter((r["entity_kind"], r["context"]) for r in rows)
    print(f"saved results/{args.out}: {len(rows)} items ({len(rows)//4} quadruples)")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")


if __name__ == "__main__":
    main()
