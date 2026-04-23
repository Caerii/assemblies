# Papers Organization

This directory is for paper writing that emerges from validated research in
`../claims/`.

It is not the place to decide what the science is. That should already be
settled, or at least sharply bounded, before a draft starts here.

## Principle

Papers come after claims, not before them.

The intended flow is:

1. question
2. experiment
3. result
4. claim
5. paper

## Layout

```text
papers/
|-- _latex_infrastructure/
|-- _shared_assets/
`-- drafts/
```

### `_latex_infrastructure/`

Shared build tooling, templates, preambles, and style support for paper
writing.

### `_shared_assets/`

Reusable figures, tables, equations, bibliography entries, and other assets
that should not be copied independently into each draft.

### `drafts/`

One directory per active paper draft.

## Starting a Paper

Before creating a draft:

- check `../claims/index.json`
- identify the exact claims or evidence summaries the paper is built around
- verify that the claim scope is narrow enough to defend

If the claim is still moving quickly, it is usually too early for a paper
draft.

## Draft Structure

A typical draft directory should contain:

```text
drafts/<paper_name>/
|-- main.tex
|-- sections/
|-- figures/
|-- notes/
`-- Makefile
```

Use shared assets wherever possible instead of duplicating them into each
paper.

## Build Workflow

For LaTeX setup details, editor configuration, and build tooling, see
`_latex_infrastructure/README.md`.

Typical commands inside a paper draft are:

```bash
make quick
make full
make watch
```

Use quick builds while writing and full builds before review or submission.

## Writing Rules

- keep the draft tied to explicit claims
- cite the underlying papers for literature claims
- do not let the paper state more strongly than `claims/` does
- keep figure/table sources reusable in `_shared_assets/`
- prefer one sentence per line for cleaner diffs

## Current Status

This directory is infrastructure-first right now. It should stay relatively
light until more claims are promoted into publishable form.
