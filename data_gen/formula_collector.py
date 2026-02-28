"""
Formula collector: generates LaTeX formulas from templates and loads public datasets.

Covers math, chemistry, and physics formula types across three complexity levels.
"""

import random
import re
import argparse
import json
from pathlib import Path
from typing import Iterator


GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta",
    r"\eta", r"\theta", r"\iota", r"\kappa", r"\lambda", r"\mu",
    r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau",
    r"\upsilon", r"\phi", r"\chi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
    r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega",
]

VARIABLES = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
OPERATORS = ["+", "-", r"\cdot", r"\times", r"\div", r"\pm", r"\mp"]
RELATIONS = ["=", r"\neq", r"\leq", r"\geq", "<", ">", r"\approx", r"\equiv", r"\sim"]
FUNCTIONS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp", r"\sqrt"]

CHEM_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Fe", "Cu", "Zn", "Ag", "Au", "Pt", "Hg", "Pb", "U",
]


def _rand_var() -> str:
    return random.choice(VARIABLES + GREEK)


def _rand_digit() -> str:
    return str(random.randint(1, 99))


def _rand_op() -> str:
    return random.choice(OPERATORS)


def _rand_rel() -> str:
    return random.choice(RELATIONS)


def _rand_func() -> str:
    return random.choice(FUNCTIONS)


def _rand_subscript(base: str) -> str:
    sub = random.choice([_rand_digit(), _rand_var()])
    return f"{base}_{{{sub}}}"


def _rand_superscript(base: str) -> str:
    sup = random.choice([_rand_digit(), _rand_var()])
    return f"{base}^{{{sup}}}"


# ---------------------------------------------------------------------------
# Simple formulas
# ---------------------------------------------------------------------------

def gen_simple_arithmetic() -> str:
    a, b = _rand_var(), _rand_var()
    op = _rand_op()
    rel = _rand_rel()
    c = _rand_var()
    return f"{a} {op} {b} {rel} {c}"


def gen_simple_fraction() -> str:
    a, b = _rand_var(), _rand_var()
    return rf"\frac{{{a}}}{{{b}}}"


def gen_simple_power() -> str:
    base = _rand_var()
    exp = random.choice([_rand_digit(), _rand_var()])
    return f"{base}^{{{exp}}}"


def gen_simple_sqrt() -> str:
    content = random.choice([
        _rand_var(),
        f"{_rand_var()} {_rand_op()} {_rand_var()}",
    ])
    if random.random() < 0.3:
        n = random.randint(3, 5)
        return rf"\sqrt[{n}]{{{content}}}"
    return rf"\sqrt{{{content}}}"


def gen_simple_subscript() -> str:
    return _rand_subscript(_rand_var())


# ---------------------------------------------------------------------------
# Medium formulas
# ---------------------------------------------------------------------------

def gen_medium_integral() -> str:
    var = random.choice(["x", "t", "u", "s"])
    integrand_parts = [_rand_var()]
    if random.random() < 0.5:
        integrand_parts.append(f"^{{{_rand_digit()}}}")
    integrand = "".join(integrand_parts)

    if random.random() < 0.5:
        a, b = _rand_var(), _rand_var()
        return rf"\int_{{{a}}}^{{{b}}} {integrand} \, d{var}"
    return rf"\int {integrand} \, d{var}"


def gen_medium_sum() -> str:
    idx = random.choice(["i", "j", "k", "n", "m"])
    lower = f"{idx}={random.choice(['0', '1'])}"
    upper = random.choice(["n", "N", r"\infty"])
    body = random.choice([
        _rand_var() + f"_{{{idx}}}",
        rf"\frac{{{_rand_var()}_{{{idx}}}}}{{{_rand_digit()}}}",
    ])
    return rf"\sum_{{{lower}}}^{{{upper}}} {body}"


def gen_medium_product() -> str:
    idx = random.choice(["i", "j", "k"])
    lower = f"{idx}=1"
    upper = random.choice(["n", "N"])
    body = _rand_var() + f"_{{{idx}}}"
    return rf"\prod_{{{lower}}}^{{{upper}}} {body}"


def gen_medium_limit() -> str:
    var = random.choice(["x", "n", "t"])
    target = random.choice(["0", r"\infty", "a"])
    body = random.choice([
        rf"\frac{{{_rand_func()}{{{var}}}}}{{{var}}}",
        rf"\left( 1 + \frac{{1}}{{{var}}} \right)^{{{var}}}",
    ])
    return rf"\lim_{{{var} \to {target}}} {body}"


def gen_medium_matrix() -> str:
    rows = random.randint(2, 3)
    cols = random.randint(2, 3)
    env = random.choice(["pmatrix", "bmatrix", "vmatrix"])
    lines = []
    for _ in range(rows):
        row = " & ".join(_rand_var() for _ in range(cols))
        lines.append(row)
    body = r" \\ ".join(lines)
    return rf"\begin{{{env}}} {body} \end{{{env}}}"


def gen_medium_binomial() -> str:
    n = _rand_var()
    k = _rand_var()
    return rf"\binom{{{n}}}{{{k}}}"


# ---------------------------------------------------------------------------
# Complex formulas
# ---------------------------------------------------------------------------

def gen_complex_nested_fraction() -> str:
    a, b, c, d = (_rand_var() for _ in range(4))
    return rf"\frac{{\frac{{{a}}}{{{b}}} + {c}}}{{\frac{{{d}}}{{{a}}} - {b}}}"


def gen_complex_multiline() -> str:
    lines = []
    for _ in range(random.randint(2, 4)):
        lhs = f"{_rand_var()} {_rand_op()} {_rand_var()}"
        rhs = _rand_var()
        lines.append(f"{lhs} &{_rand_rel()} {rhs}")
    body = r" \\ ".join(lines)
    return rf"\begin{{aligned}} {body} \end{{aligned}}"


def gen_complex_series() -> str:
    var = random.choice(["x", "z"])
    idx = "n"
    coeff = _rand_var()
    return rf"\sum_{{n=0}}^{{\infty}} \frac{{{coeff}_{{n}} {var}^{{n}}}}{{n!}}"


def gen_complex_derivative() -> str:
    order = random.choice(["", "2", "n"])
    func = _rand_var()
    var = random.choice(["x", "t"])
    if order == "":
        return rf"\frac{{d{func}}}{{d{var}}}"
    return rf"\frac{{d^{{{order}}} {func}}}{{d{var}^{{{order}}}}}"


def gen_complex_pde() -> str:
    func = random.choice(["u", "\\psi", "\\phi"])
    return rf"\frac{{\partial^2 {func}}}{{\partial x^2}} + \frac{{\partial^2 {func}}}{{\partial y^2}} = 0"


def gen_complex_cases() -> str:
    n_cases = random.randint(2, 4)
    lines = []
    for i in range(n_cases):
        expr = f"{_rand_var()} {_rand_op()} {_rand_digit()}"
        cond = random.choice([
            rf"\text{{if }} {_rand_var()} > 0",
            rf"\text{{if }} {_rand_var()} = {_rand_digit()}",
            rf"\text{{otherwise}}",
        ])
        lines.append(f"{expr} & {cond}")
    body = r" \\ ".join(lines)
    return rf"{_rand_var()} = \begin{{cases}} {body} \end{{cases}}"


# ---------------------------------------------------------------------------
# Chemistry formulas
# ---------------------------------------------------------------------------

def _rand_element(with_count: bool = True) -> str:
    el = random.choice(CHEM_ELEMENTS)
    if with_count and random.random() < 0.5:
        n = random.randint(2, 12)
        return f"\\text{{{el}}}_{{{n}}}"
    return f"\\text{{{el}}}"


def gen_chem_simple() -> str:
    reactants = [_rand_element() for _ in range(random.randint(1, 3))]
    products = [_rand_element() for _ in range(random.randint(1, 3))]
    arrow = random.choice([r"\rightarrow", r"\longrightarrow", r"\xrightarrow{\Delta}"])
    return f"{' + '.join(reactants)} {arrow} {' + '.join(products)}"


def gen_chem_equilibrium() -> str:
    reactants = [_rand_element() for _ in range(random.randint(1, 2))]
    products = [_rand_element() for _ in range(random.randint(1, 2))]
    return f"{' + '.join(reactants)} \\rightleftharpoons {' + '.join(products)}"


def gen_chem_ion() -> str:
    el = random.choice(CHEM_ELEMENTS[:20])
    charge = random.choice(["", "2", "3"])
    sign = random.choice(["+", "-"])
    return f"\\text{{{el}}}^{{{charge}{sign}}}"


# ---------------------------------------------------------------------------
# Physics formulas
# ---------------------------------------------------------------------------

def gen_physics_einstein() -> str:
    return r"E = mc^{2}"


def gen_physics_schrodinger() -> str:
    return r"i\hbar \frac{\partial}{\partial t} \Psi = \hat{H} \Psi"


def gen_physics_maxwell() -> str:
    eqs = [
        r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}",
        r"\nabla \cdot \mathbf{B} = 0",
        r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
        r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}",
    ]
    return random.choice(eqs)


def gen_physics_kinematics() -> str:
    templates = [
        r"v = v_0 + at",
        r"x = x_0 + v_0 t + \frac{1}{2}at^{2}",
        r"v^{2} = v_0^{2} + 2a(x - x_0)",
        r"F = ma",
        r"p = mv",
        r"E_k = \frac{1}{2}mv^{2}",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "simple": [
        gen_simple_arithmetic,
        gen_simple_fraction,
        gen_simple_power,
        gen_simple_sqrt,
        gen_simple_subscript,
    ],
    "medium": [
        gen_medium_integral,
        gen_medium_sum,
        gen_medium_product,
        gen_medium_limit,
        gen_medium_matrix,
        gen_medium_binomial,
    ],
    "complex": [
        gen_complex_nested_fraction,
        gen_complex_multiline,
        gen_complex_series,
        gen_complex_derivative,
        gen_complex_pde,
        gen_complex_cases,
    ],
    "chemistry": [
        gen_chem_simple,
        gen_chem_equilibrium,
        gen_chem_ion,
    ],
    "physics": [
        gen_physics_einstein,
        gen_physics_schrodinger,
        gen_physics_maxwell,
        gen_physics_kinematics,
    ],
}

COMPLEXITY_WEIGHTS = {
    "simple": 0.30,
    "medium": 0.30,
    "complex": 0.20,
    "chemistry": 0.10,
    "physics": 0.10,
}


def generate_formulas(count: int, seed: int = 42) -> Iterator[dict]:
    """Yield dicts with keys: latex, category, complexity."""
    random.seed(seed)
    categories = list(COMPLEXITY_WEIGHTS.keys())
    weights = [COMPLEXITY_WEIGHTS[c] for c in categories]

    for i in range(count):
        cat = random.choices(categories, weights=weights, k=1)[0]
        gen_fn = random.choice(GENERATORS[cat])
        latex = gen_fn()
        yield {
            "id": f"synth_{i:07d}",
            "latex": latex,
            "category": cat,
        }


def load_im2latex(data_dir: str) -> Iterator[dict]:
    """Load formulas from im2latex-230k format files."""
    formulas_path = Path(data_dir) / "im2latex_formulas.norm.lst"
    if not formulas_path.exists():
        print(f"Warning: {formulas_path} not found. Run scripts/download_data.sh first.")
        return

    formulas = formulas_path.read_text(encoding="utf-8").strip().split("\n")

    for split_name in ["train", "validate", "test"]:
        lst_path = Path(data_dir) / f"im2latex_{split_name}_filter.lst"
        if not lst_path.exists():
            continue
        for line in lst_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            formula_idx = int(parts[0])
            image_name = parts[1]
            if formula_idx < len(formulas):
                yield {
                    "id": f"im2latex_{image_name}",
                    "latex": formulas[formula_idx].strip(),
                    "category": "external",
                    "split": split_name,
                    "image_name": image_name + ".png",
                }


def main():
    parser = argparse.ArgumentParser(description="Generate or load LaTeX formulas")
    parser.add_argument("--mode", choices=["generate", "load", "both"], default="generate")
    parser.add_argument("--count", type=int, default=10000, help="Number of synthetic formulas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-dir", type=str, default="data/external")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        if args.mode in ("generate", "both"):
            for item in generate_formulas(args.count, args.seed):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

        if args.mode in ("load", "both"):
            for item in load_im2latex(args.external_dir):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} formulas to {output_path}")


if __name__ == "__main__":
    main()
