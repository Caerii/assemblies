"""First investigation: does learned recurrence recover a partial cue?

Run: uv run python examples/01_basic_assembly_calculus.py
Instructional demonstration, not preregistered evidence or a capacity claim.
"""
from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus import project, pattern_complete
from neural_assemblies.diagnostics import ensemble, paired_delta


def recovery(seed: int, beta: float) -> float:
    # Fixed graph, Binomial stimulus counts, lowest-index ties. No lazy sampler.
    brain = Brain(engine='numpy_exact', seed=seed, p=.1, norm_init=False)
    brain.add_stimulus('cue', 30)
    brain.add_area('memory', n=1000, k=30, beta=beta)
    project(brain, 'cue', 'memory', rounds=12, recurrent=True)
    with brain.read_only():
        _, score = pattern_complete(brain, 'memory', fraction=.5, rounds=5, seed=seed)
    return score

def main():
    seeds = [1, 2, 3]
    null = ensemble(lambda seed: recovery(seed, beta=0), seeds, 'learning disabled')
    trained = ensemble(lambda seed: recovery(seed, beta=.2), seeds, 'recurrent training')
    print('DEMONSTRATION: fixed-connectome CPU, not preregistered scientific evidence')
    print(null)
    print(trained)
    print(paired_delta(trained, null, 'training minus null'))

if __name__ == "__main__":
    main()
