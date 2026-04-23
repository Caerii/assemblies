# Project Context

This document preserves the longer context behind the repository without making
the root README carry all of it.

It is the right place for project history, motivation, and positioning details
that matter for understanding why the repo exists, but which should not be
mistaken for package-level guarantees.

For package-backed claims, read [scientific_status.md](scientific_status.md).
For package-versus-legacy boundaries, read
[supported_surfaces.md](supported_surfaces.md).

## Why This Repo Exists

This repository is part implementation, part research workspace.

The package exists to make assembly-calculus ideas runnable, inspectable, and
extensible in software. The broader repo exists because the package is only one
piece of the work: there are experiments, historical prototypes, language
systems, archived artifacts, and research questions that do not belong in the
default package contract.

## Origins

The work in this repository grew out of assembly-calculus and language-organ
research around MIT's Projects in the Science of Intelligence course and later
extensions of that line of work.

The current repository has a specific human provenance that should not be lost
when the docs get tighter:

- Alif Jakir has been the main builder, extender, and curator of this repo in
  its current form.
- The work connects directly to collaboration with Daniel Mitropolsky
  (MIT Poggio Lab) and the broader assembly-calculus / language-organ research
  line.
- The package and research tree should therefore be read as part of a concrete
  research lineage, not as an anonymous software artifact that appeared from
  nowhere.

This matters because the repository is not just a generic implementation of an
idea. It reflects specific technical choices, research priorities, and
iterations made while extending that line of work.

One important part of the repo's history is that it did not start as a
package-first codebase. It accumulated through research iterations, prototypes,
and exploratory experiments, and was later reorganized into a cleaner package,
archive, and research structure.

## Historical Milestones

### 2024: Extending the assembly-calculus line

An early motivating question was whether assembly-calculus and NEMO-style ideas
could support more than the most standard toy demonstrations, including visual
discrimination tasks.

That phase of the work includes Alif Jakir's extension of Daniel
Mitropolsky's thesis-oriented assembly-calculus and language-organ direction
into a broader experimental codebase, including exploratory visual
discrimination work.

That led to exploratory visual experiments, including CIFAR-oriented work. The
important historical lesson from those experiments is:

- the approach showed feasibility in principle
- it did not yet establish robust category formation at the level that should
  be advertised as a package guarantee

That is why the current docs treat those results as research history rather
than productized capability.

### Rewrites for clarity

The repository was rewritten several times to make it easier to extend and
understand. A major engineering theme has been moving from a loose
checkout-driven experiment repository toward:

- an installable package under `neural_assemblies/`
- archived historical surfaces under `legacy/`
- explicit research workflow under `research/`

### 2025: Scaling and accelerator work

Another major effort was scaling the system through custom CUDA and related
engineering work. That work matters for the trajectory of the project, but the
right current claim is still modest:

- GPU and accelerator work are real and important here
- exact speedups and large-scale guarantees depend on workload and environment
- packaging and docs should describe the current tested surface, not the most
  ambitious aspiration

## Research Motivation

The broader motivation behind the repository is that assemblies might offer a
different substrate for computation than standard deep-learning pipelines:

- sparse and interpretable intermediate structure
- local learning rules such as Hebbian updates
- compositional operations over assemblies rather than only next-token-style
  statistical prediction
- a route to language, memory, and structured computation that stays closer to
  the motivating neuroscience literature

That motivation is real context for the repo, but it should be read as a
research program and design direction, not as proof that every hoped-for
capability has already been achieved.

## Positioning

Several points from the older README are still worth keeping, just not in the
landing page.

### This is an implementation and experiment platform

The repo implements real runtime and package surfaces, but it also carries
research code and historical baggage. It should not be read as a polished proof
artifact for every theoretical claim in the surrounding literature.

### This is not the only assembly-calculus implementation

The repository is one implementation with its own extensions and engineering
choices, including NEMO-related code, lexicon/curriculum infrastructure,
accelerator work, and a large research tree.

### "Co-built with AI" needs interpretation

The repository has been developed with AI assistance, but that should be read
as an engineering-process note, not as a claim that scientific credit or design
ownership is diffuse or ambiguous. The intended meaning is:

- AI assisted with coding, organization, and iteration
- scientific direction and curation remain deliberate human choices led here by
  Alif Jakir, within a research context that also involves collaboration with
  Daniel Mitropolsky

## Intended Audience

The repo is mainly for:

- researchers and students in computational neuroscience
- people exploring neuro-inspired alternatives to mainstream deep learning
- readers interested in assembly calculus, NEMO-style language work, and sparse
  structured computation

It is less appropriate for someone looking for:

- a generic ML production framework
- a hosted API
- a biophysical simulator
- a package that claims to have already solved language or AGI

## How To Use This Context Correctly

Use this file to understand the repository's intent and history.

Do not use this file as the strongest evidence source for concrete capability
claims. For that:

- package-backed claims -> [scientific_status.md](scientific_status.md)
- repo structure and boundaries -> [supported_surfaces.md](supported_surfaces.md)
- experiments and evidence -> [../research/README.md](../research/README.md)
- indexed claims -> [../research/claims/index.json](../research/claims/index.json)
- curated scientific questions -> [../research/core_questions/index.json](../research/core_questions/index.json)
