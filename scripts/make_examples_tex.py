"""Build docs/examples.tex: HarmBench prompts (under the prefill attack) where BLADE (rho=.002)
refuses but ActAdd/CAA/RefDir all comply (Kimi-K3 judged). BLADE refusal excerpt shown; steering
baselines shown as a short (non-actionable) compliance opening. Pick rows for the paper."""
import json
from pathlib import Path

R = Path("results"); OUT = Path("docs/examples.tex")
lab = json.loads((R / "harmbench_peritem_labels.json").read_text())
gens = json.loads((R / "harmbench_gens_for_judge.json").read_text())
B = "BLADE_L14_r002_a1.5"
OTHERS = [("ActAdd_L14_c8", "ActAdd"), ("CAA_L20_c0.5", "CAA"), ("RefDir_L16_c0.35", "RefDir")]
IDX = [69, 81, 85, 88, 92, 102, 105, 106, 109, 142, 144, 156, 170]


def esc(s):
    s = " ".join(s.split())
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ('"', "''")]:
        s = s.replace(a, b)
    return s


def clip(s, n):
    s = " ".join(s.split())
    return esc(s[:n].rstrip()) + ("\\dots" if len(s) > n else "")


rows = []
for i in IDX:
    req = clip(gens[B]["items"][i - 1]["request"], 90)
    blade = clip(gens[B]["items"][i - 1]["continuation"], 150)
    # one representative steering compliance opening (CAA), kept short + non-actionable
    caa = clip(gens["CAA_L20_c0.5"]["items"][i - 1]["continuation"], 70)
    rows.append((i, req, blade, caa))

lines = [
    r"% Qualitative examples: BLADE (rho=.002) refuses; ActAdd(c=8), CAA(c=.5), RefDir(c=.35) all comply.",
    r"% Under a prefill attack (responses forced to begin ``Sure, here is''). Kimi-K3 judged. Pick rows.",
    r"\begin{table*}[t]",
    r"  \centering\small",
    r"  \caption{\textbf{Under a prefill attack, BLADE recovers refusal where activation steering does"
    r" not.} HarmBench requests on which BLADE amplification ($\rho{=}0.002$) is judged to refuse while"
    r" all three steering baselines (ActAdd $c{=}8$, CAA $c{=}0.5$, RefDir $c{=}0.35$) comply (Kimi-K3)."
    r" Responses are forced to begin ``Sure, here is''; BLADE reverses into a refusal, the baselines"
    r" continue. Steering text is truncated to a non-actionable opening.}",
    r"  \label{tab:refusal-examples}",
    r"  \begin{tabularx}{\textwidth}{@{}p{0.24\textwidth} X p{0.24\textwidth}@{}}",
    r"    \toprule",
    r"    \textbf{Harmful request} & \textbf{BLADE ($\rho{=}0.002$) --- refuses} &"
    r" \textbf{Steering (ActAdd/CAA/RefDir) --- comply} \\",
    r"    \midrule",
]
for i, req, blade, caa in rows:
    lines.append(f"    {req} & \\emph{{{blade}}} & all comply, e.g.\\ CAA: \\emph{{{caa}}} \\\\")
    lines.append(r"    \addlinespace[2pt]")
lines += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table*}"]
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} with {len(rows)} rows")
