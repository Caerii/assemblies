# Literature PDFs

Downloaded arXiv PDFs for figure/table extraction. Run:

```bash
uv run python research/literature/extract_figure_notes.py
```

Output: [../FIGURES.md](../FIGURES.md), [../figure_snippets.json](../figure_snippets.json)

| File | Paper |
|------|-------|
| `hoff2026_epwta.pdf` | E%-WTA inhibition (Hoff et al. 2026) |
| `kopadi2026_direct.pdf` | DIRECT causal learning (Kopadi & Kalles 2026) |
| `dabagia2024_coinflipping.pdf` | Coin-flipping statistical learning |
| `mitropolsky2025_acquisition.pdf` | Simulated language acquisition |
| `dabagia2025_sequences.pdf` | Sequences / FSM / TM |
| `dabagia2022_colt.pdf` | COLT classification |
| `ting2026_speech.pdf` | Speech segmentation AC |

| `papadimitriou2019_itcs.pdf` | ITCS 2019 random projection |
| `mitropolsky2021_tacl.pdf` | TACL 2021 parser |
| `papadimitriou2020_pnas.pdf` | PNAS 2020 (PMC if paywalled) |

PNAS 2020: if local file is tiny, use [PMC7459898](https://pmc.ncbi.nlm.nih.gov/articles/PMC7459898/) or DOI in [index.json](../index.json).
