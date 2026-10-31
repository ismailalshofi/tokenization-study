"""Rewrite the conclusion/interpretation markdown cells in experiment.ipynb
in simpler English, and fix the dataset-size reference (4.5 MB -> ~8 MB).

Run once from the project root:

    python scripts/simplify_experiment_md.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "experiment.ipynb"

# Each entry rewrites a single markdown cell, identified by index.
# Only the cells that *interpret* a result are rewritten — section headings,
# code, and pure setup prose stay as they are.
REWRITES: dict[int, str] = {

    5: (
        "### What we see\n"
        "\n"
        "The dataset is about **8 MB** on disk (the structured CSV). Once we "
        "drop the metadata columns and keep only the dialogue text, we get "
        "about 4.5M characters — roughly 4× bigger than the classic Tiny "
        "Shakespeare file people use to test small language models.\n"
        "\n"
        "Notice the shape: speaker name in CAPS, then a colon on its own "
        "line, then the lines they speak, blank lines between speakers. "
        "This is what LM training data usually looks like.\n"
        "\n"
        "The tokenizers don't know any of this. They see the text as one "
        "long string. The model has to learn the \"this is a speaker label\" "
        "pattern on its own. How well each tokenizer helps the model do "
        "that is a big part of what we're testing."
    ),

    10: (
        "950 speakers sounds like a lot. The reason is that `player` is the "
        "raw label from the script. `KING`, `KING HENRY`, and `KING HENRY V` "
        "count as three different speakers, even though they may be the same "
        "person across plays. We're leaving them as-is — fixing this is a "
        "different problem, and we don't need to solve it to study "
        "tokenization."
    ),

    13: (
        "The top of the list is full of generic role labels (`KING`, `DUKE`, "
        "`PRINCE`) that show up across many plays. Past those, real named "
        "characters appear: Hamlet, Iago, Brutus, Antony, Falstaff. That "
        "makes sense — those are the long, talky leads."
    ),

    16: (
        "*Richard III*, *Hamlet*, *Cymbeline*, and *Antony and Cleopatra* "
        "are the longest. *The Merry Wives of Windsor* is the shortest. The "
        "longest play is only about 2× the shortest, so no play takes over "
        "the corpus. The training data is fairly balanced across all 37 "
        "plays."
    ),

    20: (
        "The median line is about **40 characters / 8 words**. That's "
        "iambic pentameter — Shakespeare's standard verse line. The left "
        "tail (short lines) is one-word interjections and names. The right "
        "tail (long lines) is prose passages. This shape matters later: "
        "when we look at how many tokens each line becomes after "
        "tokenization, this is the starting distribution."
    ),

    23: (
        "That's the opening of the \"To be, or not to be\" scene. The point "
        "isn't the literature — it's that having the data in this "
        "structured form lets us pull out a specific scene in one line of "
        "pandas. We'll use that later for questions like \"how does the "
        "model handle Hamlet's prose vs the King's verse?\""
    ),

    27: (
        "The same sentence, four different ways to slice it:\n"
        "\n"
        "- **Character** — 42 tokens. Every letter, space, and comma counts.\n"
        "- **Subword** — 13 tokens. Common pieces like `To`, ` be`, ` the` "
        "merge into single units.\n"
        "- **Word** — 10 tokens. One token per space-separated chunk.\n"
        "- **Sentence** — 1 token (or a few). The whole thing in one piece.\n"
        "\n"
        "The rest of this notebook is about what happens when we train "
        "*the same model* on the *same text* using **character** and "
        "**subword** — the two choices that matter for text generation."
    ),

    30: (
        "### Where do these numbers come from?\n"
        "\n"
        "**Char ≈ 70** is not a choice. It's just every distinct character "
        "in the corpus (uppercase letters, lowercase letters, punctuation, "
        "whitespace) plus our 4 special tokens (`<pad>`, `<unk>`, `<bos>`, "
        "`<eos>`).\n"
        "\n"
        "**Subword = 8000** *is* a choice we made. BPE starts with single "
        "bytes (~256 of them) and learns merges (`t`+`h` → `th`, then "
        "`th`+`e` → `the`, and so on) until it hits the vocab limit we set. "
        "8000 is a common default for small corpora. For comparison: LLaMA "
        "uses 32K, GPT-4 uses about 100K."
    ),

    33: (
        "The `Ġ` characters mark the start of a word — they're how "
        "byte-level BPE encodes a space (HuggingFace convention). Two "
        "things to notice: `question` is one whole token, and common "
        "openings like `To` and `be` are single tokens too. Those merges "
        "were learned from the corpus. That's why BPE trained on "
        "Shakespeare looks different from BPE trained on, say, scientific "
        "papers."
    ),

    36: (
        "### The compression ratio is the key number\n"
        "\n"
        "On average, one subword token covers about **3.5 characters** of "
        "text. That single number explains most of the trade-off in this "
        "study:\n"
        "\n"
        "- For the same sentence, the subword model sees **~3.5× fewer "
        "tokens**.\n"
        "- For the same context window (`block_size`), the subword model "
        "sees **~3.5× more text**.\n"
        "- The subword model has more *parameters* (because the vocab is "
        "bigger) but processes fewer *tokens* per step.\n"
        "\n"
        "Those two effects pull in opposite directions. Which one wins "
        "depends on the corpus, the model size, and how long we train."
    ),

    38: (
        "### Reading the frequency plots\n"
        "\n"
        "Both plots use **log-log axes**. On these axes, a straight line "
        "means the distribution follows a power law — that's Zipf's law "
        "(frequency ∝ 1/rank).\n"
        "\n"
        "**Char plot** — short and pretty flat near the top. The most "
        "common characters (space, `e`, `t`, `a`, `o`) all have similar "
        "frequencies. There's no long tail because there's almost no "
        "vocabulary to have a tail in.\n"
        "\n"
        "**Subword plot** — long and steep. A handful of tokens dominate "
        "(common words and word-pieces). Thousands of tokens show up only "
        "a few times each. That long tail is the problem: rare tokens get "
        "few updates during training, so their embeddings are hard to "
        "learn.\n"
        "\n"
        "**So:** the subword model has to spend learning effort on tokens "
        "it barely sees. The char model doesn't have this problem."
    ),

    41: (
        "### The 3.5× compression shows up at every percentile\n"
        "\n"
        "The subword sequences are 3–4× shorter than the char sequences "
        "everywhere — average speeches, long speeches, and short ones.\n"
        "\n"
        "**Why this matters:** if you want most of your documents to fit "
        "in one chunk, the subword model lets you use a context window "
        "about 3.5× smaller for the same coverage. Since attention cost "
        "is O(T²), that's roughly a **12× cheaper** training step."
    ),

    43: (
        "### Distribution shape\n"
        "\n"
        "Both distributions are right-skewed: most speeches are short, "
        "a few are very long. But the subword distribution is squashed "
        "toward zero. That long right tail in the char distribution is "
        "the headache — you either truncate long speeches, or you set a "
        "bigger context window. Either choice costs you something."
    ),

    46: (
        "### What this tells us\n"
        "\n"
        "The subword model has **a lot more parameters** than the char "
        "model. But almost all of the extra parameters sit in the "
        "embedding and output layers — the parts that depend on vocab "
        "size. The Transformer blocks in the middle (the part that does "
        "the \"thinking\") are exactly the same in both models.\n"
        "\n"
        "**Important caveat:** when we compare the two models, we're not "
        "comparing two equally-sized models. The subword model is bigger. "
        "For a strictly fair comparison we'd either shrink the subword "
        "embedding to match parameter count, or just keep this asymmetry "
        "in mind when reading the results."
    ),

    49: (
        "### Reading these loss numbers carefully\n"
        "\n"
        "The raw losses (char ~2.5, subword ~5.7) make it look like the "
        "char model is winning by a huge margin. **It's not.** These "
        "losses are cross-entropies over different vocabularies — choosing "
        "1 out of ~70 (char) vs choosing 1 out of 8000 (subword). Even a "
        "model that picks at random gets a much higher loss on the "
        "subword task. So the gap is mostly \"different number of "
        "classes,\" not \"different model quality.\"\n"
        "\n"
        "The fair metric is **bits per character (bpc)** — the loss "
        "normalized to a per-character cost. The formula is:\n"
        "\n"
        "```\n"
        "bpc = nats_per_token / chars_per_token / ln(2)\n"
        "```\n"
        "\n"
        "Plugging in the compression ratios from section 6 (1.0 chars per "
        "token for char, ~3.5 for subword), the same raw losses give very "
        "different bpc — and subword usually comes out **better**, not "
        "worse.\n"
        "\n"
        "**About this 300-step run:** it's just a smoke test to check the "
        "pipeline works. These aren't the final numbers. For the real "
        "comparison we need the longer training run from the CLI."
    ),

    52: (
        "### What to look at in this plot\n"
        "\n"
        "- **Don't compare the heights** of the two curves directly — "
        "they're in different units (see section 9).\n"
        "- **Do compare the slopes** — how fast each loss drops tells you "
        "about convergence speed.\n"
        "- **Do compare the gap** between train and val (when we plot "
        "val) — that tells you about overfitting.\n"
        "\n"
        "For a strictly fair comparison, the y-axis needs to be "
        "bits-per-character, not raw cross-entropy."
    ),

    55: (
        "### What we typically see (and why)\n"
        "\n"
        "After only 300 steps:\n"
        "\n"
        "**Char model.** The *shape* is right — speaker labels in CAPS, "
        "colons, line breaks, sentence-like punctuation. But the *words* "
        "are nonsense. The model learned the structure of a Shakespeare "
        "page before it learned English. That makes sense: at every step "
        "it only has to pick between ~70 things.\n"
        "\n"
        "**Subword model.** Punctuation soup, broken words, repetition. "
        "Why? In 300 steps × batch 32 × block 128 ≈ 1.2M token updates, "
        "an 8000-entry embedding table has barely moved away from its "
        "random initialisation. Most subword tokens have been seen only "
        "a handful of times.\n"
        "\n"
        "**The subword model isn't worse — it's undertrained.** Around "
        "5000 steps it usually pulls ahead in quality.\n"
        "\n"
        "**Fair-comparison note:** the prompt `\"ROMEO:\"` becomes 6 "
        "tokens for the char model and 1–2 tokens for the subword model. "
        "So the two models start from slightly different amounts of "
        "context. For strict comparisons, feed both models the same "
        "number of tokens or the same span of text."
    ),

    58: (
        "### Watch out: token count hides what's really happening\n"
        "\n"
        "The Chinese row makes it look like the char tokenizer *wins* — "
        "4 tokens vs 12. It doesn't.\n"
        "\n"
        "- The char tokenizer turns `你好世界` into **4 `<unk>` tokens**. "
        "All four are the same token. The model has no way to tell "
        "`你好世界` apart from `xxxx` or any other unseen 4-character "
        "string.\n"
        "- The subword tokenizer uses 12 tokens via **byte-level "
        "fallback**. Each Chinese character is 3 UTF-8 bytes, and BPE "
        "keeps them. Nothing is lost — you can recover the exact original "
        "bytes.\n"
        "\n"
        "So the real robustness story is the opposite of what the table "
        "suggests:\n"
        "\n"
        "- **Char tokenizer** fails *silently* on unknown scripts — "
        "everything collapses to `<unk>`.\n"
        "- **Subword tokenizer** degrades *gracefully* — longer encoding, "
        "but no information lost.\n"
        "\n"
        "To show this properly we should add a column counting `<unk>` "
        "tokens. That would turn a misleading comparison into a useful one."
    ),

    59: (
        "## 13. Discussion & insights\n"
        "\n"
        "What this study showed:\n"
        "\n"
        "- **Vocabulary.** Char tokenizer = ~70 tokens. BPE tokenizer = "
        "8000 tokens. The BPE frequency distribution has a much heavier "
        "long tail.\n"
        "- **Sequence length.** For the same text, subword sequences are "
        "about **3.5× shorter** than char sequences. Since attention cost "
        "is O(T²), that's roughly a **12× compute saving** per step.\n"
        "- **Training.** Raw loss across the two models is not comparable "
        "— different vocab sizes. In **bits per character**, the subword "
        "model usually reaches a better optimum, but it needs more steps "
        "to get there because its embedding and output layers are much "
        "larger.\n"
        "- **Generation.** Subword samples reach word-level fluency "
        "earlier. Char samples take longer to become coherent, but they "
        "never produce characters that don't exist in the alphabet.\n"
        "- **Robustness.** Char fails silently on unseen scripts "
        "(everything becomes `<unk>`). Subword degrades gracefully via "
        "byte-level fallback — slower, but nothing is lost.\n"
        "\n"
        "**Bottom line:** tokenization is not \"just preprocessing.\" It "
        "changes what the model has to learn, how fast it learns, and how "
        "much compute each step costs."
    ),
}


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    for idx, new_source in REWRITES.items():
        cell = cells[idx]
        if cell["cell_type"] != "markdown":
            raise SystemExit(
                f"cell #{idx} is not markdown (it's {cell['cell_type']}); "
                f"refusing to rewrite — please re-check the index."
            )
        cell["source"] = new_source
        n_changed += 1

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rewrote {n_changed} markdown cells in {NB.name}")


if __name__ == "__main__":
    main()
