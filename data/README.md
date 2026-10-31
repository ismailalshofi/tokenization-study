# Shakespeare for NLP: 91K Tagged Lines

A structured CSV of **every spoken line** from all **37 of Shakespeare's plays**, sourced from the **Folger Shakespeare Library** edition (via the [dracor-org/shakedracor](https://github.com/dracor-org/shakedracor) TEI XML mirror) and parsed into a one-row-per-line tabular format — ready for tokenizer training, speaker classification, stylometry, RAG demos, and small-scale LM pretraining.

| Stat | Value |
| --- | --- |
| Plays | 37 |
| Spoken lines | 91,509 |
| Unique speakers | 956 |
| File size | ~8.9 MB |
| Format | UTF-8 CSV |

## Schema

| Column        | Type   | Description |
| ------------- | ------ | --- |
| `play`        | string | Play slug, e.g. `hamlet`, `romeo-and-juliet` |
| `play_title`  | string | Human-readable title, e.g. `Hamlet`, `Romeo and Juliet` |
| `act`         | int    | Act number (1–5) |
| `scene`       | int    | Scene number within the act |
| `player`      | string | Speaker name in uppercase, e.g. `HAMLET`, `LADY MACBETH` |
| `line_number` | string | Folger reference, e.g. `1.1.2` (act.scene.line). Empty for partial / shared lines that don't anchor a full verse line. |
| `line_text`   | string | The spoken text (stage directions removed) |

## Example

```
play     play_title  act  scene  player     line_number  line_text
hamlet   Hamlet      1    1      BARNARDO                Who's there?
hamlet   Hamlet      1    1      FRANCISCO  1.1.2        Nay, answer me. Stand and unfold yourself.
hamlet   Hamlet      1    1      BARNARDO                Long live the King!
hamlet   Hamlet      1    1      FRANCISCO               Barnardo.
hamlet   Hamlet      1    1      BARNARDO                He.
hamlet   Hamlet      1    1      FRANCISCO  1.1.6        You come most carefully upon your hour.
```

## Source & lineage

- **Edition:** Folger Shakespeare (the Folger Library's modern critical edition).
- **Format:** TEI Simple XML.
- **Mirror:** [dracor-org/shakedracor](https://github.com/dracor-org/shakedracor) on GitHub.
- **Parser:** a custom Python script that recursively walks `<div type="act">` → `<div type="scene">` → `<sp>` → `<l>`/`<p>`, dropping stage directions and normalising whitespace.

## License

The underlying Folger text is released by the Folger Shakespeare Library under **CC BY-NC 3.0** — *free for non-commercial use with attribution*. This Kaggle re-release inherits that license; please cite the Folger Shakespeare in any derived work.

> Folger Shakespeare Library. *Shakespeare's Plays, Sonnets, and Poems*. Edited by Barbara Mowat, Paul Werstine, Michael Poston, and Rebecca Niles. <https://folgerdigitaltexts.org>

## Suggested uses

- **Language-model training corpora** — long-form, stylistically consistent, public, well-known.
- **Tokenization experiments** — character vs subword, vocabulary-size scaling, OOV/robustness probes.
- **Speaker classification** — predict speaker from a line, conditioned on play/act/scene.
- **Stylometry / network analysis** — speakers as nodes, co-scene appearances as edges (the original dracor-org use case).
- **Information retrieval / RAG demos** — well-known passages, easy to ground-truth.

## Notes & caveats

- **Smart punctuation preserved** — apostrophes and dashes use real Unicode characters (`'`, `—`) as in the Folger edition.
- **Empty `line_number`** — short or shared lines that don't anchor a full poetic line have no Folger reference. The row order within `(play, act, scene)` reflects the original document order.
- **`player` is a raw label**, not a canonical character ID. Some characters appear under multiple labels across plays (e.g. `KING`, `KING HENRY`, `KING HENRY V`) — normalise downstream if you need a consistent identity.
- **Stage directions** are *not* included; only spoken text.
