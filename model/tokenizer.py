"""
LaTeX tokenizer: regex-based tokenizer with a vocabulary covering
math, chemistry, physics symbols, and common LaTeX commands.

The tokenizer splits LaTeX into meaningful tokens (commands, symbols,
braces, operators) and maps them to integer IDs.
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

LATEX_TOKEN_PATTERN = re.compile(
    r"(\\(?:[a-zA-Z]+|[^a-zA-Z\s]))"  # LaTeX commands: \frac, \alpha, \{, etc.
    r"|([a-zA-Z])"                      # single letters
    r"|([0-9]+)"                        # numbers
    r"|(\s+)"                           # whitespace
    r"|([{}()\[\]|&=+\-*/^_!<>,.:;])"  # structural and operator characters
    r"|(.)"                             # any remaining character
)

CORE_VOCAB = [
    # structural
    "{", "}", "(", ")", "[", "]", "^", "_", "&", "|",
    # operators and relations
    "+", "-", "*", "/", "=", "<", ">", ",", ".", ":", ";", "!",
    # common LaTeX commands - math
    r"\frac", r"\dfrac", r"\tfrac", r"\sqrt", r"\sum", r"\prod",
    r"\int", r"\iint", r"\iiint", r"\oint",
    r"\lim", r"\sup", r"\inf", r"\max", r"\min",
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh",
    r"\log", r"\ln", r"\exp",
    r"\partial", r"\nabla", r"\infty",
    r"\to", r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftarrow",
    r"\longrightarrow", r"\longleftarrow",
    r"\leftrightarrow", r"\Leftrightarrow",
    r"\mapsto",
    r"\forall", r"\exists", r"\nexists",
    r"\in", r"\notin", r"\subset", r"\supset", r"\subseteq", r"\supseteq",
    r"\cup", r"\cap", r"\setminus", r"\emptyset",
    r"\wedge", r"\vee", r"\neg",
    r"\cdot", r"\times", r"\div", r"\pm", r"\mp",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim", r"\simeq", r"\cong",
    r"\propto", r"\perp", r"\parallel",
    r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\overline", r"\underline", r"\hat", r"\bar", r"\tilde", r"\vec", r"\dot", r"\ddot",
    r"\overbrace", r"\underbrace",
    r"\binom", r"\choose",
    r"\text", r"\mathrm", r"\mathbf", r"\mathit", r"\mathcal", r"\mathbb", r"\mathfrak",
    r"\boldsymbol",
    # Greek letters
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\zeta", r"\eta", r"\theta", r"\vartheta",
    r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi",
    r"\pi", r"\varpi", r"\rho", r"\varrho",
    r"\sigma", r"\varsigma", r"\tau", r"\upsilon",
    r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
    r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
    # environments
    r"\begin", r"\end",
    "aligned", "cases", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix",
    "matrix", "array", "equation", "gather",
    r"\\",  # line break
    # spacing
    r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad",
    # chemistry
    r"\rightleftharpoons", r"\xrightarrow",
    r"\ce",  # mhchem package
    # physics
    r"\hbar", r"\ell",
    r"\mathbf",
    # accents and decorations
    r"\prime", "'",
    # misc
    r"\ldots", r"\cdots", r"\vdots", r"\ddots",
    r"\dagger", r"\ddagger",
    r"\circ", r"\bullet", r"\star",
]


def tokenize_latex(latex: str) -> list[str]:
    """Split a LaTeX string into a list of tokens."""
    tokens = []
    for match in LATEX_TOKEN_PATTERN.finditer(latex):
        token = match.group(0)
        if token.isspace():
            if token == " ":
                tokens.append(" ")
            continue
        tokens.append(token)
    return tokens


class LaTeXTokenizer:
    """Bidirectional mapping between LaTeX tokens and integer IDs."""

    def __init__(self, vocab: Optional[dict[str, int]] = None):
        if vocab is not None:
            self.token2id = dict(vocab)
        else:
            self.token2id = dict(SPECIAL_TOKENS)
            idx = len(SPECIAL_TOKENS)
            for token in CORE_VOCAB:
                if token not in self.token2id:
                    self.token2id[token] = idx
                    idx += 1
            for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch not in self.token2id:
                    self.token2id[ch] = idx
                    idx += 1
            for d in "0123456789":
                if d not in self.token2id:
                    self.token2id[d] = idx
                    idx += 1
            if " " not in self.token2id:
                self.token2id[" "] = idx

        self.id2token = {v: k for k, v in self.token2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token2id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token2id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token2id["<unk>"]

    def encode(self, latex: str, add_special: bool = True) -> list[int]:
        """Convert LaTeX string to list of token IDs."""
        tokens = tokenize_latex(latex)
        ids = []
        if add_special:
            ids.append(self.bos_id)
        for token in tokens:
            ids.append(self.token2id.get(token, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert list of token IDs back to LaTeX string."""
        tokens = []
        for token_id in ids:
            token = self.id2token.get(token_id, "<unk>")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def build_from_corpus(self, formulas: list[str], min_freq: int = 2) -> None:
        """Extend vocabulary from a corpus of LaTeX formulas."""
        counter = Counter()
        for formula in formulas:
            tokens = tokenize_latex(formula)
            counter.update(tokens)

        idx = max(self.token2id.values()) + 1
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = idx
                idx += 1

        self.id2token = {v: k for k, v in self.token2id.items()}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LaTeXTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


def main():
    parser = argparse.ArgumentParser(description="Build LaTeX tokenizer vocabulary")
    parser.add_argument("--corpus", type=str, help="JSONL file with 'latex' field")
    parser.add_argument("--output", type=str, required=True, help="Output vocab JSON path")
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()

    tokenizer = LaTeXTokenizer()
    print(f"Core vocab size: {tokenizer.vocab_size}")

    if args.corpus:
        formulas = []
        with open(args.corpus, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                formulas.append(item["latex"])
        print(f"Building vocab from {len(formulas)} formulas...")
        tokenizer.build_from_corpus(formulas, min_freq=args.min_freq)
        print(f"Extended vocab size: {tokenizer.vocab_size}")

    tokenizer.save(args.output)
    print(f"Saved vocabulary to {args.output}")

    test_formulas = [
        r"\frac{1}{2}",
        r"\int_{0}^{\infty} e^{-x^2} dx",
        r"\sum_{n=0}^{\infty} \frac{x^n}{n!}",
        r"\text{H}_2\text{O}",
    ]
    print("\nTokenization examples:")
    for formula in test_formulas:
        tokens = tokenize_latex(formula)
        ids = tokenizer.encode(formula)
        decoded = tokenizer.decode(ids)
        print(f"  {formula}")
        print(f"    tokens: {tokens}")
        print(f"    ids:    {ids}")
        print(f"    decode: {decoded}")
        print()


if __name__ == "__main__":
    main()
