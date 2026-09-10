# A1 measurement sensitivity to disabling learning

Registered before the null run. The trained arm has already scored perfectly in
historical replay; this is an instrument validation, not a new capacity claim.

Use the configuration from the committed horizon-record-consumed-20260910/run.json:
20 seeds 1..20, p=.3 and .4, 2000 digits per seed, n_arc5000,n_state500,k70,
15 presentations, beta.1,strength.1,clip20,no initial normalization. Fresh
HashedArcFSM objects in both arms; no learned state or disk model is reused.
The null changes only beta=0 and strength=0, disabling potentiation and bias.
Teacher-forced state assignment and the training schedule remain. Input digit
streams remain random.Random(seed*7919), paired exactly. Run one GPU job at a time.
Order is trained/null at p=.3 and null/trained at p=.4. This is not a speed study.

Before seeing null data, require ALL these bars independently at BOTH p:

- trained accuracy's 95% lower bound > .99;
- null accuracy's 95% upper bound < .90;
- paired trained-minus-null accuracy's 95% lower bound > .10;
- paired trained-minus-null exact-state fraction's 95% lower bound > .10.

Use the existing ensemble_from_values and paired_delta Student-t intervals over
20 brain seeds. Timesteps are not independent samples; probabilities are not
pooled. These are nominal marginal intervals, not simultaneous confidence bounds.
Retain every seed's raw metrics, intervals, criteria and failed bars. A perfect
null must FAIL. A smoke run shortens length to50 and has VOID scientific status.
No threshold, seed set or protocol change after seeing data may be adopted here.

This tests the combined mechanism, not its separate components. Passing validates
measurement sensitivity in this fixture; it does not prove neural implementation
correctness, biological plausibility or a new scientific theorem. Execution uses
python -m research.runner a1-learning-null --tag UNIQUE. Results belong under
research/results/runs/sequence.a1-learning-null/ with automatic run provenance.
