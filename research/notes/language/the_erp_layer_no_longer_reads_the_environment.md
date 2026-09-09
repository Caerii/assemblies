# `erp/adapters.py` has no `import os` (#115)

The ERP measurement layer now reads no environment variable at all. Every choice
that changes what an ERP number MEANS arrives as an `ErpProtocol` argument.

`protocol.py` predicted this and stated why it was deferred:

> The call chain from `measure_live_integration` down to `phrase_stability` is
> NOT yet threaded -- that touches the live ERP metric, which is under active
> investigation (#108) and whose magnitudes must not move as a side effect of a
> plumbing change. **Thread it behind a measurement, not with one.**

#108 closed at `f79c4f5` and the afferent-energy question with it at `0fcedb4`,
so the condition was met.

## The three env reads, and why they were not equally bad

**`phrase_stability` — the worst placed.** It runs once per phrase area per
probe, so an `os.environ` read there made the environment a *per-area* input to
a measurement three call levels below anyone who chose it. It now takes a
**required** `protocol` with no default. A default of `from_environment()` would
have looked threaded while leaving the door open — the shape rule 3 in
[[one-canonical-way]] is about: prefer making the wrong way unspellable.

**`_expected_slot_enabled()` — deleted.** A zero-argument function whose answer
comes from `os.environ` is exactly the shape `protocol.py` exists to remove. Its
long docstring was a research record, and records do not belong on a function
that should not exist; it lives in
`research/notes/language/p600_the_honest_number_is_0717.md` and, condensed, on
`ErpProtocol.expected_slot`.

**`_ERP_DEBUG` — the one that was actually broken.** A MODULE-LEVEL read,
evaluated at import. `ERP_DEBUG=1` set by any test or study *after* the module
loaded did nothing whatsoever, and said nothing about it. It is now
`protocol.debug`, resolved per call. This is the only behavioural change in the
commit, and it can only affect printing.

## Where the environment is read now

Two boundaries, both of which resolve once and pass the value down:

| entry point | resolves | passes to |
|---|---|---|
| `measure_live_integration` | `protocol or from_environment()` | `phrase_stability` |
| `measure_lexical_surprise` | same | its own debug print |
| `run_incremental_erp_probes` | same, once per PARSE | every probe in that parse |
| `collect_frame_samples` | same, once per FRAME SET | every probe |
| `trial_category_in_sentence` | same, once per TRIAL | both ERP reads |

The drivers matter as much as the leaves. Resolving once per parse means **one
parse can no longer measure its first word under one protocol and its last under
another** — which a per-probe environment read permitted, and which no test
would ever have caught because both readings return a number.

## What this is not

It is not a numerical change. The same values are resolved from the same
environment; the default `ErpProtocol()` is the shipped protocol. The claim
being made is about *attributability*: a result can now be traced to an arm from
the call site, because the arm arrived as an argument rather than as a property
of when you looked.

Related: [[one-canonical-way]], [[same-name-two-meanings]].
