# Offline RAG answer evaluation

Run `venv/bin/python scripts/evaluate_answers.py`. This evaluation uses the
synthetic source fixture and passes explicitly selected context through the
production `RAGService`, with a deterministic scripted text provider. It makes
no network requests and does not establish real model quality.

## Metrics and interpretation

- **Expected-fact exact-match proxy** counts expected fact phrases found in
  scripted answers after case folding and whitespace normalization. The 8/9
  aggregate includes eight evidence-supported expected facts plus one
  intentionally withheld fact (“Room 4”). It measures lexical coverage of all
  listed expected facts, not overall answer correctness. The supported-fact
  subset is reported separately as 8/8. Neither metric evaluates entailment.
- **Fact-support annotation validation** distinguishes `supported` facts with
  one or more supplied chunk IDs from `withheld` facts with no evidence IDs.
  Validation confirms internal references and status consistency; it cannot
  prove that a source semantically entails a fact.
- **Evidence-support judgments** are checked-in claim annotations grouped as
  supported, contradicted, or unsupported. These are fixture annotations, not
  judgments of model outputs. The fixture records no reviewer identity or
  review date, so human-review provenance cannot be independently verified.
- **Retrieved-source attribution precision/recall** compare complete source
  tuples `(document_id, chunk_id, filename, chunk_index)` returned by
  `RAGService` with the case's expected source tuple set, macro-averaged across
  cases. `RAGService` cites every retrieved chunk, not only chunks used by each
  answer claim. Extra retrieved sources can therefore lower attribution
  precision even when the scripted answer omits them. This is not claim-level
  citation correctness. Missing citations reduce recall; unexpected citations
  reduce precision; metadata mismatches count as non-matching tuples. Duplicate
  actual or expected tuples are rejected. Empty actual citations have precision
  1 by convention; recall is 0 if citations were expected. Two empty sets score
  1 for both, and unexpected-only citations with no expected sources score
  precision 0 and recall 1.
- **Unsupported-claim exact-phrase check** flags a declared claim when its
  normalized full phrase occurs in the answer. It can detect the checked-in
  scripted negative case, but is only a lexical check and does not establish
  semantic groundedness.
- **Abstention phrase-match proxy** reports whether a configured phrase appears
  in an unanswerable answer. The fixture includes an explicit false positive:
  an answer says the notes do not say, then makes an unsupported assertion.
  Phrase presence is not verified abstention.
- **Retrieval ranking** remains separate in `scripts/evaluate_retrieval.py`.

## Optional records for reviewed real outputs

Optional records may be stored in `evaluation/reviews/records.json` using this
shape (the directory/file need only be created when a reviewed record exists):

```json
{
  "version": 1,
  "records": [{
    "case_id": "release-date",
    "model_label": "locally recorded model/version",
    "answer": "The release is May 12.",
    "evidence": [{"chunk_id":"chunk-release","document_id":"doc-project","filename":"project-plan.txt","chunk_index":0,"text":"The Northstar release is scheduled for May 12."}],
    "judgments": {"supported":["The release is May 12."],"contradicted":[],"unsupported":[]},
    "rationale": "Date is stated in supplied evidence.",
    "reviewer": "initials-or-local-reviewer-id",
    "reviewed_at": "2026-10-22"
  }]
}
```

Create a record only after a reviewer inspects the exact answer and supplied
evidence. Copy only synthetic or otherwise approved, non-sensitive material;
omit credentials and unrelated private documents. These optional records are
not consumed by the default runner. Unreviewed model outputs must not be
treated as ground truth.

## Reproducible scripted baseline

- Cases: 8.
- Expected-fact exact-match proxy: 8/9 (0.889), including the intentionally
  withheld fact in its denominator.
- Supported-fact exact-match proxy: 8/8 (1.000).
- Checked-in judgment annotation entries: 6 supported, 0 contradicted,
  1 unsupported. These describe scripted fixture annotations, not model runs;
  review provenance is not recorded.
- Retrieved-source attribution precision: 0.938 macro; recall: 1.000 macro.
- Unsupported-claim exact-phrase matches: 1/2 declared claims.
- Abstention phrase-match proxy: 3/3 unanswerable cases, including one known
  false positive that continues with an unsupported assertion.

These results verify fixture plumbing and limited mechanical checks only. They
are not a Cohere run, model benchmark, semantic groundedness rate, or overall
answer-correctness score. The synthetic set is small and cannot represent real
document variety.
