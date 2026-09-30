"""Quick tier experiment benchmark."""
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_ventral_bundle
from neural_assemblies.programs.colt_mnist_tier_a import run_merge_halves_mnist, run_multi_prototype_mnist
from neural_assemblies.programs.colt_mnist_tier_b import run_consolidation_mnist, run_pattern_completion_mnist
from neural_assemblies.programs.colt_mnist_lri_readout import run_lri_cascade_mnist
from neural_assemblies.programs.cross_domain_assemblies import run_vision_language_digit_hub

clear_ventral_bundle_cache()
kw = dict(seed=42, n_examples=50)
b = load_ventral_bundle(**kw)
print("multi:", f"{run_multi_prototype_mnist(bundle=b, **kw).mean_accuracy:.1%}")
mh = run_merge_halves_mnist(bundle=b, **kw)
print("merge:", f"{mh.mean_accuracy:.1%}", "fidelity", mh.extra.get("routing_fidelity"))
p = run_pattern_completion_mnist(bundle=b, **kw)
print("pcomp:", f"{p.mean_accuracy:.1%}", "rec", f"{p.mean_recovery:.3f}")
print("consol:", f"{run_consolidation_mnist(bundle=b, **kw).mean_accuracy:.1%}")
l = run_lri_cascade_mnist(bundle=b, **kw)
print("lri:", f"{l.mean_accuracy:.1%}", "base", l.extra.get("baseline_readout_accuracy"), "delta", l.extra.get("confused_digit_delta"))
x = run_vision_language_digit_hub(bundle=b, **kw)
print("cross:", f"agree={x.agreement_rate:.1%} vis={x.visual_accuracy:.1%} lang={x.language_accuracy:.1%}")
