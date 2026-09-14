# MAX

MAX is an evidence-oriented hybrid Arabic understanding core. It separates
morphology, structural syntax, semantic interpretation, learned observations,
and persistent memory. It treats unknown terms as unknown and persists
observations without promoting them to facts. Conflicting learning evidence is
retained as a hypothesis instead of replacing existing learned evidence.

`LearnExample` is the supervised-training boundary: it aligns a labelled Arabic
clause structurally, records the labelled episode, and learns only its lexical
role evidence. It validates every role before mutation, so a conflicting example
cannot leave partially learned state behind.

Run the evaluation suite with `go test ./...`.

The suite includes a generated 500,000-clause Arabic composition evaluator. It
learns only 250 word-level observations, stores no training sentences, and then
checks 500,000 unique VSO/SVO clauses. This is a regression guard for structural
generalisation; it is not a claim of broad natural-language understanding.
