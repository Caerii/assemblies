# Volume 3: Language

This volume uses controlled examples to show how assembly operations support
word categories, roles, and word order.

The guiding idea is simple: a word is not only a string. In these examples it
can have a category, land in a role area, and participate in a learned order.

Use the notebooks here as package demonstrations, not as claims that the system
solves language acquisition. Stronger claims should link to tests, experiments,
or papers.

## Notebooks

- `01_nemo_parser_toy_sentence.ipynb`: train the maintained
  `assembly_calculus.NemoParser` on one toy SVO sentence and inspect the parse.

Notes:

- The optional `neural_assemblies.nemo` stack currently requires optional GPU
  dependencies such as CuPy. Do not make beginner notebooks depend on it until
  that path has a cleaner CPU fallback.
- The older rule-based parser path is useful legacy material, but it is not the
  right first teaching surface.

Good future notebooks in this volume should add one concept at a time: more
sentences, then novel word order tests, then failure cases. Avoid jumping
straight from a toy parser to broad claims about language.
