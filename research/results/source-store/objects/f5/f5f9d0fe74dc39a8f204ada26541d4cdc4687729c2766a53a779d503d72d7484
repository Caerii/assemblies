from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache, load_ventral_bundle
from neural_assemblies.programs.colt_mnist_tier_a import run_merge_halves_mnist
from neural_assemblies.programs.colt_mnist_tier_b import run_consolidation_mnist

clear_ventral_bundle_cache()
kw = dict(seed=42, n_examples=50)
b = load_ventral_bundle(**kw)
print("merge:", f"{run_merge_halves_mnist(bundle=b, **kw).mean_accuracy:.1%}")
c = run_consolidation_mnist(bundle=b, **kw)
print("consol:", f"{c.mean_accuracy:.1%}", "pre", c.extra.get("pre_replay_accuracy"), "delta", c.extra.get("delta_vs_pre_replay"))
