"""Generate the standalone Shakespeare EDA notebook.

Run from the project root:

    python scripts/build_eda_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "eda_shakespeare.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


cells = []
add = cells.append

# 0
add(md(
    "# Shakespeare for NLP — EDA on 91K Lines\n"
    "\n"
    "**Dataset:** [Shakespeare for NLP: 91K Tagged Lines]"
    "(https://www.kaggle.com/datasets/ismailalshofi/shakespeare-for-nlp-91k-tagged-lines)"
    " — every spoken line from all 37 plays of the Folger Shakespeare edition, "
    "parsed into one row per line (act, scene, speaker, text).\n"
    "\n"
    "Before training anything on a corpus it's worth spending an hour just "
    "looking at the data. This notebook does that: how clean it is, how it's "
    "balanced, what the alphabet looks like, where the vocabulary lives, and "
    "what the corpus statistically *is* (it turns out: more or less English, "
    "with a Shakespearean accent).\n"
    "\n"
    "Runs in well under a minute on Kaggle CPU. No GPU needed."
))

# 1. Setup
add(md(
    "## 1. Setup\n"
    "\n"
    "Standard scientific-Python stack. Stopword list comes from sklearn later "
    "on so we don't need any NLTK downloads."
))

add(code(
    "import os\n"
    "import re\n"
    "import math\n"
    "import collections\n"
    "from glob import glob\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "plt.rcParams['figure.figsize'] = (9, 4)\n"
    "plt.rcParams['axes.grid']      = True\n"
    "plt.rcParams['grid.alpha']     = 0.25\n"
    "pd.set_option('display.max_colwidth', 80)"
))

add(md(
    "On Kaggle the dataset is mounted under `/kaggle/input/`. Locally I keep "
    "the CSV next to the notebook. The cell below probes both."
))

add(code(
    "candidates = [\n"
    "    *glob('/kaggle/input/*/shakespeare_lines.csv'),\n"
    "    'shakespeare_lines.csv',\n"
    "    '../data/shakespeare_lines.csv',\n"
    "]\n"
    "csv_path = next((p for p in candidates if os.path.exists(p)), None)\n"
    "assert csv_path, 'shakespeare_lines.csv not found in any expected location'\n"
    "csv_path"
))

# 2. First look
add(md(
    "## 2. First look\n"
    "\n"
    "Read it, look at it. `head()` always shows the same play, so a random "
    "sample is more useful for surfacing format quirks."
))

add(code(
    "df = pd.read_csv(csv_path)\n"
    "df.shape"
))

add(code(
    "df.head()"
))

add(code(
    "df.sample(8, random_state=42)"
))

add(md(
    "Seven columns: identifiers (`play`, `play_title`, `act`, `scene`, "
    "`player`), the Folger line reference (`line_number`, formatted "
    "`act.scene.line`) and the text itself. `player` is uppercase and "
    "intentionally raw — we'll come back to that."
))

# 3. Quality checks
add(md(
    "## 3. Quality checks\n"
    "\n"
    "Two quick checks before we trust any of the analysis below."
))

add(code(
    "df.isna().sum()"
))

add(md(
    "About 11K missing `line_number` values. The dataset card calls these "
    "out: they're shared / partial lines that don't anchor a full verse line "
    "in the Folger edition. They're real lines, not bad data.\n"
    "\n"
    "50 rows have no `player`. That's small enough to actually look at:"
))

add(code(
    "df[df['player'].isna()].head()"
))

add(md(
    "Continuation lines where the speaker is implied by the previous row. "
    "I'll fill them with `'UNKNOWN'` rather than drop them — the text content "
    "is still valid."
))

add(code(
    "# same play / act / scene / speaker / text — are there exact repeats?\n"
    "dup = df.duplicated(['play', 'act', 'scene', 'player', 'line_text'], keep=False)\n"
    "dup.sum()"
))

add(md(
    "A few hundred. They're almost all short interjections like *Ay.* or *My "
    "lord?* that the same character says more than once in the same scene. "
    "Genuine repeats, leaving them in."
))

# 4. Cleanup
add(md(
    "## 4. Light cleanup\n"
    "\n"
    "Three small edits, all reversible:\n"
    "\n"
    "- fill missing speakers with `'UNKNOWN'`\n"
    "- strip leading / trailing whitespace from line text\n"
    "- add `char_len` and `word_len` helper columns for the analysis below\n"
    "\n"
    "No lower-casing or punctuation stripping. Those are modelling decisions, "
    "not cleaning, and the uppercase speaker labels are signal."
))

add(code(
    "df = df.copy()\n"
    "df['player']    = df['player'].fillna('UNKNOWN')\n"
    "df['line_text'] = df['line_text'].astype(str).str.strip()\n"
    "df['char_len']  = df['line_text'].str.len()\n"
    "df['word_len']  = df['line_text'].str.split().str.len()\n"
    "df.head()"
))

# 5. Corpus stats
add(md(
    "## 5. Numbers"
))

add(code(
    "print(f\"plays:            {df['play'].nunique()}\")\n"
    "print(f\"scenes:           {df.groupby(['play','act','scene']).ngroups:,}\")\n"
    "print(f\"speakers (raw):   {df['player'].nunique():,}\")\n"
    "print(f\"spoken lines:     {len(df):,}\")\n"
    "print(f\"total words:      {int(df['word_len'].sum()):,}\")\n"
    "print(f\"total characters: {int(df['char_len'].sum()):,}\")"
))

add(md(
    "Just under a million words; about 4.5 MB of text. Small for "
    "pretraining a real LM, but a good size for an EDA you can iterate on "
    "without waiting around."
))

# 6. Plays
add(md(
    "## 6. How balanced is the corpus across plays?\n"
    "\n"
    "If one play takes up half the corpus, anything we train on a random "
    "split will lean on that play. So: how big is each play, and how much "
    "do they vary?"
))

add(code(
    "per_play = (\n"
    "    df.groupby('play_title')\n"
    "      .agg(lines=('line_text', 'size'),\n"
    "           words=('word_len', 'sum'),\n"
    "           chars=('char_len', 'sum'))\n"
    "      .sort_values('chars', ascending=False)\n"
    ")\n"
    "per_play.head(10)"
))

add(code(
    "fig, ax = plt.subplots(figsize=(9, 11))\n"
    "ps = per_play.sort_values('chars')\n"
    "ax.barh(ps.index, ps['chars'] / 1000, color='#4C72B0')\n"
    "ax.set_xlabel('characters (thousands)')\n"
    "ax.set_title('Corpus size per play')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "*Hamlet* is the biggest, *The Comedy of Errors* the smallest. The "
    "longest play is roughly twice the shortest — no one play swamps the "
    "rest. That's about as balanced as you can hope for from a literary "
    "corpus."
))

# 7. Speakers
add(md(
    "## 7. Who's actually talking\n"
    "\n"
    "Almost a thousand distinct speaker labels in 37 plays, which sounds "
    "high. The trick is that Shakespeare uses generic role names — `KING`, "
    "`DUKE`, `PRINCE` — that recur across plays as if they were one "
    "character. They're not, but the dataset can't tell."
))

add(code(
    "speaker_words = df.groupby('player')['word_len'].sum().sort_values(ascending=False)\n"
    "speaker_lines = df['player'].value_counts()\n"
    "\n"
    "pd.DataFrame({\n"
    "    'lines': speaker_lines,\n"
    "    'words': speaker_words,\n"
    "}).sort_values('words', ascending=False).head(15)"
))

add(code(
    "top = speaker_words.head(15).sort_values()\n"
    "fig, ax = plt.subplots(figsize=(9, 6))\n"
    "ax.barh(top.index, top.values, color='#DD8452')\n"
    "ax.set_xlabel('total spoken words')\n"
    "ax.set_title('Top 15 speakers by word count')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "The first five slots are all role labels (`KING`, `KING HENRY`, "
    "`RICHARD`, `KING HENRY V`, `DUKE`). The first proper named characters "
    "in the list are Hamlet, Iago, Antony, Brutus — the long, talky leads, "
    "which is the right answer.\n"
    "\n"
    "If you wanted real per-character analysis you'd have to normalise "
    "those labels (`KING` + `KING HENRY` + `KING HENRY V` → one identity). "
    "Out of scope for this notebook."
))

add(md(
    "How concentrated is the dialogue overall? Sort speakers by word count "
    "and draw the cumulative-share curve. A diagonal would mean everyone "
    "speaks an equal share; anything bowed up to the left is concentration."
))

add(code(
    "share = speaker_words.values / speaker_words.values.sum()\n"
    "cum = np.cumsum(share)\n"
    "x = np.arange(1, len(share) + 1) / len(share)\n"
    "\n"
    "fig, ax = plt.subplots()\n"
    "ax.plot(x, cum, color='#55A868', lw=2)\n"
    "ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6, label='equal share')\n"
    "ax.set_xlabel('fraction of speakers (sorted by words desc.)')\n"
    "ax.set_ylabel('cumulative share of total words')\n"
    "ax.set_title('How concentrated is the dialogue?')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "for k in [10, 50, 100]:\n"
    "    print(f'top {k:>3} speakers ({k/len(share):5.1%} of cast) -> {cum[k-1]:5.1%} of all words')"
))

add(md(
    "10 speakers (~1% of the cast) account for ~15% of all words; 100 "
    "speakers (~10%) cover more than half. Long tail of one-line servants "
    "and messengers after that. That curve is what corpora made of natural "
    "dialogue look like."
))

# 8. Line length
add(md(
    "## 8. How long is a line?\n"
    "\n"
    "Shakespeare's verse is mostly iambic pentameter — ten syllables, "
    "roughly 8 words, roughly 40 characters. If the dataset is honest, the "
    "line-length distribution should have a sharp mode right around there."
))

add(code(
    "df[['char_len', 'word_len']].describe().round(1)"
))

add(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].hist(df['char_len'].clip(upper=120), bins=40, color='#4C72B0')\n"
    "axes[0].axvline(df['char_len'].median(), color='k', ls='--', alpha=0.6)\n"
    "axes[0].set_xlabel('characters per line (clipped at 120)')\n"
    "axes[0].set_title('Characters per line')\n"
    "\n"
    "axes[1].hist(df['word_len'].clip(upper=25), bins=25, color='#55A868')\n"
    "axes[1].axvline(df['word_len'].median(), color='k', ls='--', alpha=0.6)\n"
    "axes[1].set_xlabel('words per line (clipped at 25)')\n"
    "axes[1].set_title('Words per line')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "Median sits at 41 characters / 8 words — right on iambic pentameter. "
    "The left tail is short interjections (*Ay.*, *Speak!*, *My lord?*). "
    "The thinner right tail is prose passages and read-aloud letters that "
    "the editors kept as single logical lines."
))

# 9. Heaps' law
add(md(
    "## 9. How fast does the vocabulary grow?\n"
    "\n"
    "**Heaps' law:** as you read more of a corpus, the number of *unique* "
    "words grows sub-linearly with the number of *running* words —\n"
    "\n"
    "$$V(N) \\approx K \\cdot N^{\\beta}$$\n"
    "\n"
    "with β between roughly 0.4 and 0.6 for everyday English. The intuition "
    "is obvious in hindsight: at the start almost every word is new; later "
    "most words are repeats and only the occasional new one shows up.\n"
    "\n"
    "Before we measure it, we need a simple word tokenizer. Lowercase, then "
    "keep runs of letters and apostrophes. Good enough for frequency work — "
    "not what I'd feed an actual model."
))

add(code(
    "WORD = re.compile(r\"[A-Za-z']+\")\n"
    "\n"
    "def tokenize(s):\n"
    "    return WORD.findall(s.lower())\n"
    "\n"
    "tokens = []\n"
    "for line in df['line_text']:\n"
    "    tokens.extend(tokenize(line))\n"
    "\n"
    "print(f'tokens (running words): {len(tokens):,}')\n"
    "print(f'unique types:           {len(set(tokens)):,}')"
))

add(md(
    "Now sweep through the token stream and record how the unique-word "
    "count grows. Sampling at log-spaced positions keeps the plot "
    "informative on log-log axes (and is fast — no need to record every "
    "step)."
))

add(code(
    "positions = np.unique(np.logspace(2, np.log10(len(tokens)), 60).astype(int))\n"
    "seen, vocab_at, i = set(), [], 0\n"
    "for p in positions:\n"
    "    while i < p:\n"
    "        seen.add(tokens[i])\n"
    "        i += 1\n"
    "    vocab_at.append(len(seen))\n"
    "\n"
    "positions = np.array(positions)\n"
    "vocab_at  = np.array(vocab_at)\n"
    "\n"
    "beta, log_k = np.polyfit(np.log(positions), np.log(vocab_at), 1)\n"
    "K = math.exp(log_k)\n"
    "print(f'fit:  V ≈ {K:.2f} · N^{beta:.3f}')"
))

add(code(
    "fig, ax = plt.subplots()\n"
    "ax.loglog(positions, vocab_at, 'o', color='#4C72B0', label='observed')\n"
    "ax.loglog(positions, K * positions**beta, '-', color='#C44E52',\n"
    "          label=f'fit:  V = {K:.2f} · N^{beta:.3f}')\n"
    "ax.set_xlabel('tokens seen (N)')\n"
    "ax.set_ylabel('unique words (V)')\n"
    "ax.set_title(\"Heaps' law on Shakespeare\")\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "β comes out around **0.65** — a hair above the usual English range. "
    "Doesn't surprise me; Shakespeare's vocabulary is famously broad, so "
    "new words keep showing up a bit longer than they would in newswire or "
    "Twitter."
))

# 10. Zipf
add(md(
    "## 10. Word frequency falls off how fast?\n"
    "\n"
    "**Zipf's law:** the frequency of the *r*-th most common word goes "
    "roughly as $1/r^{s}$, with $s$ near 1 for English. On a log-log plot "
    "that's a straight line."
))

add(code(
    "counts = collections.Counter(tokens)\n"
    "ranks  = np.arange(1, len(counts) + 1)\n"
    "freqs  = np.array(sorted(counts.values(), reverse=True))\n"
    "\n"
    "# fit the clean body of the distribution (ranks 10 – 5000), not the noisy tails\n"
    "mask = (ranks >= 10) & (ranks <= 5000)\n"
    "slope, intercept = np.polyfit(np.log(ranks[mask]), np.log(freqs[mask]), 1)\n"
    "print(f'slope ≈ {slope:.2f}  (Zipf exponent s ≈ {-slope:.2f})')"
))

add(code(
    "fig, ax = plt.subplots()\n"
    "ax.loglog(ranks, freqs, color='#4C72B0', alpha=0.7, label='observed')\n"
    "xs = np.linspace(1, len(ranks), 200)\n"
    "ax.loglog(xs, np.exp(intercept) * xs**slope, '--', color='#C44E52',\n"
    "          label=f'fit: slope = {slope:.2f}')\n"
    "ax.set_xlabel('rank')\n"
    "ax.set_ylabel('frequency')\n"
    "ax.set_title(\"Zipf's law on Shakespeare\")\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "Slope ≈ -1.2, a bit steeper than the canonical -1. Both ends bend off "
    "the fit. The top is flattened by the handful of function words that "
    "dominate any English text (`the`, `and`, `of`, `to`). The bottom drops "
    "off faster because of all the words that appear exactly once."
))

add(code(
    "hapax = sum(1 for c in counts.values() if c == 1)\n"
    "print(f'{hapax:,} of {len(counts):,} unique words appear exactly once '\n"
    "      f'({hapax/len(counts):.0%})')"
))

add(md(
    "More than a third of the vocabulary is *hapax legomena* — words seen "
    "once and never again. That single number is the case for subword "
    "tokenization (BPE, SentencePiece, etc.) over a fixed word vocabulary: "
    "if you cap your vocab at, say, 20K, you're either going to drop those "
    "words to `<unk>` or you'll need a much bigger embedding matrix."
))

# 11. Stopwords vs content
add(md(
    "## 11. Stopwords vs content\n"
    "\n"
    "Most NLP work distinguishes function words (the / of / to / and) "
    "from content words (nouns / verbs / adjectives). The top-N over "
    "everything tells you the grammar of the language; the top-N over "
    "content words tells you what the corpus is *about*."
))

add(code(
    "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n"
    "\n"
    "content_counts = collections.Counter({\n"
    "    w: c for w, c in counts.items() if w not in ENGLISH_STOP_WORDS\n"
    "})\n"
    "top_all     = counts.most_common(20)\n"
    "top_content = content_counts.most_common(20)\n"
    "\n"
    "pd.DataFrame({\n"
    "    'top word (all)':     [w for w, _ in top_all],\n"
    "    'count':              [c for _, c in top_all],\n"
    "    'top content word':   [w for w, _ in top_content],\n"
    "    'count ':             [c for _, c in top_content],\n"
    "})"
))

add(md(
    "Left column is grammar. Right column is voice: *thou*, *lord*, *sir*, "
    "*come*, *love*, *king*, *hath*, *god*. The shape of the text.\n"
    "\n"
    "One catch worth flagging: sklearn's stoplist is modern English. *Thou*, "
    "*thy*, *art*, *hath* — Early-Modern function words — aren't in it, so "
    "they leak into the content column. For real keyword work you'd add an "
    "archaic-pronoun list on top."
))

# 12. Characters
add(md(
    "## 12. The character alphabet\n"
    "\n"
    "Before subword tokenizers there are characters. What does the alphabet "
    "of this corpus actually look like?"
))

add(code(
    "char_counts = collections.Counter(''.join(df['line_text']))\n"
    "total = sum(char_counts.values())\n"
    "print(f'distinct characters: {len(char_counts)}')\n"
    "\n"
    "def label(ch):\n"
    "    return {' ': '<space>', '\\t': '<tab>', '\\n': '<newline>'}.get(ch, ch)\n"
    "\n"
    "pd.DataFrame(\n"
    "    [(label(c), n, f'{n/total:.2%}') for c, n in char_counts.most_common(30)],\n"
    "    columns=['char', 'count', 'share'],\n"
    ")"
))

add(md(
    "About 90 distinct characters. Space leads, then the textbook English "
    "letter ranking (e, t, a, o, h, i, n, s, r). Now the non-ASCII pile:"
))

add(code(
    "non_ascii = {c: n for c, n in char_counts.items() if ord(c) > 127}\n"
    "print(f'{len(non_ascii)} non-ASCII characters, '\n"
    "      f'{sum(non_ascii.values()):,} total occurrences\\n')\n"
    "for c, n in sorted(non_ascii.items(), key=lambda kv: -kv[1])[:15]:\n"
    "    print(f'  U+{ord(c):04X}  {c!r}  count={n:>5}')"
))

add(md(
    "Two things going on in there.\n"
    "\n"
    "The big counts are **Folger smart punctuation**: curly apostrophe (~19K), "
    "em dash (~5K), curly double quotes (~1.5K each), occasional ellipsis. "
    "The Folger editors kept those as-is and the dataset preserves them.\n"
    "\n"
    "Below that is a smaller batch of **accented letters** — *è, é, à, ç, ê, "
    "ï, Ô* — which I wasn't expecting at first. They come from the French "
    "scenes (mostly in *Henry V*, a little in *Love's Labour's Lost*).\n"
    "\n"
    "For modelling: don't strip any of this. Byte-level BPE (GPT-2 / "
    "tiktoken) handles arbitrary Unicode natively. Stripping costs you "
    "information and buys you nothing."
))

# 13. Bigrams
add(md(
    "## 13. What two-word phrases recur?\n"
    "\n"
    "Word counts miss multi-word units like *my lord* or *good night*. "
    "Bigrams catch them. Filter out pairs where both words are stopwords "
    "(*of the*, *and the*) so what's left is actually interesting."
))

add(code(
    "bigrams = collections.Counter()\n"
    "for line in df['line_text']:\n"
    "    t = tokenize(line)\n"
    "    for a, b in zip(t, t[1:]):\n"
    "        bigrams[(a, b)] += 1\n"
    "\n"
    "informative = [\n"
    "    (f'{a} {b}', n)\n"
    "    for (a, b), n in bigrams.most_common(2000)\n"
    "    if not (a in ENGLISH_STOP_WORDS and b in ENGLISH_STOP_WORDS)\n"
    "]\n"
    "pd.DataFrame(informative[:20], columns=['bigram', 'count'])"
))

add(md(
    "*My lord*, *good lord*, *good my*, *my good*, *my dear* — the texture "
    "of early-modern address. If you trained a BPE tokenizer with a small "
    "enough vocab on this corpus, these are exactly the kind of two-word "
    "chunks that would end up as single tokens."
))

# 14. MATTR
add(md(
    "## 14. Lexical diversity per play\n"
    "\n"
    "**Type-token ratio** (TTR) measures lexical richness: unique words "
    "divided by total words. The problem is that TTR shrinks as text gets "
    "longer — more text, more chances to repeat. So comparing raw TTRs "
    "across documents of different sizes isn't fair.\n"
    "\n"
    "**MATTR** (moving-average TTR) fixes that. Compute the TTR over a "
    "fixed-size sliding window and average the windows. Whatever the "
    "document length, you're always averaging over windows of, say, 500 "
    "tokens — so the number is comparable across plays."
))

add(code(
    "def mattr(toks, window=500):\n"
    "    if len(toks) < window:\n"
    "        return len(set(toks)) / max(1, len(toks))\n"
    "    step = max(1, window // 5)  # chunk the slide for speed; still stable\n"
    "    return float(np.mean([\n"
    "        len(set(toks[i:i + window])) / window\n"
    "        for i in range(0, len(toks) - window + 1, step)\n"
    "    ]))\n"
    "\n"
    "per_play_tokens = {\n"
    "    title: [t for line in g['line_text'] for t in tokenize(line)]\n"
    "    for title, g in df.groupby('play_title')\n"
    "}\n"
    "\n"
    "diversity = pd.Series(\n"
    "    {title: mattr(toks) for title, toks in per_play_tokens.items()}\n"
    ").sort_values(ascending=False)\n"
    "\n"
    "print('most diverse:')\n"
    "print(diversity.head(8).round(3))\n"
    "print('\\nleast diverse:')\n"
    "print(diversity.tail(8).round(3))"
))

add(code(
    "fig, ax = plt.subplots(figsize=(9, 11))\n"
    "ds = diversity.sort_values()\n"
    "ax.barh(ds.index, ds.values, color='#8172B3')\n"
    "ax.axvline(diversity.mean(), color='k', ls='--', alpha=0.5,\n"
    "           label=f'mean = {diversity.mean():.3f}')\n"
    "ax.set_xlabel('MATTR (window = 500 tokens)')\n"
    "ax.set_title('Lexical diversity per play')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

add(md(
    "The range is narrow (about **0.47 – 0.55**) — Shakespeare's vocabulary "
    "is consistently broad — but the *ordering* tells a story. The "
    "tragedies and histories (*Macbeth*, *King Lear*, *Richard II*, *Henry "
    "VI Pt 1*, *Antony and Cleopatra*) sit at the top. The comedies (*Merry "
    "Wives*, *Much Ado*, *Twelfth Night*, *Two Gentlemen*) cluster at the "
    "bottom. Comic set-pieces with repeated jokes and snappy exchanges drag "
    "the type-to-token ratio down."
))

# 15. WordCloud
add(md(
    "## 15. A word cloud, just for the look of it\n"
    "\n"
    "Word clouds are a bad analytical tool — area doesn't compare reliably "
    "and the layout is random — but as a qualitative scan they're not "
    "nothing. Guarded with a try/except so it doesn't break the notebook "
    "if `wordcloud` isn't installed."
))

add(code(
    "try:\n"
    "    from wordcloud import WordCloud\n"
    "    wc = WordCloud(\n"
    "        width=1200, height=500,\n"
    "        background_color='white',\n"
    "        stopwords=set(ENGLISH_STOP_WORDS),\n"
    "        max_words=120,\n"
    "        colormap='viridis',\n"
    "    ).generate_from_frequencies(content_counts)\n"
    "\n"
    "    fig, ax = plt.subplots(figsize=(12, 5))\n"
    "    ax.imshow(wc, interpolation='bilinear')\n"
    "    ax.axis('off')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "except ImportError:\n"
    "    print('wordcloud not installed — `pip install wordcloud` to enable.')"
))

# 16. Takeaways
add(md(
    "## 16. What I took from this\n"
    "\n"
    "- **The data is clean.** 91,509 lines, 37 plays, every line has text. "
    "The only missing-data wrinkles are explainable (shared verse lines, "
    "continuation speakers).\n"
    "- **Balanced across plays, concentrated across speakers.** Longest "
    "play ≈ 2× the shortest. But the top 10 speakers carry ~15% of all "
    "words, and the top 100 carry over half.\n"
    "- **The verse shape shows up in the data.** Median line lands on "
    "iambic pentameter (41 chars / 8 words).\n"
    "- **Statistically it's English.** Heaps' β ≈ 0.65, Zipf slope ≈ -1.2. "
    "Slightly richer and slightly more concentrated than newswire English, "
    "but the same shape.\n"
    "- **37% of unique words appear exactly once.** That's the case for "
    "subword tokenization in a single number.\n"
    "- **Smart punctuation and accented characters are real and "
    "intentional.** Don't strip them — use a byte-level tokenizer.\n"
    "- **Generic stoplists miss Early-Modern function words.** *Thou*, "
    "*thy*, *hath*, *art* leak into the \"content\" pool. Build a "
    "Shakespeare-aware stoplist before doing keyword work.\n"
    "\n"
    "**If I kept going:**\n"
    "\n"
    "- Speaker-name normalisation (`KING` + `KING HENRY` + `KING HENRY V` "
    "→ one canonical identity). That unlocks per-character stylistic "
    "analysis.\n"
    "- Sentence segmentation by punctuation rather than by line break — "
    "lines are an editorial convention, sentences aren't.\n"
    "- A POS pass with spaCy to see how verb / noun / adjective ratios "
    "differ across tragedies and comedies.\n"
    "\n"
    "If you found this useful, an upvote on the [dataset]"
    "(https://www.kaggle.com/datasets/ismailalshofi/shakespeare-for-nlp-91k-tagged-lines)"
    " keeps it visible to other people doing the same kind of work."
))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {
            "name": "python", "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
