import Std.Tactic

/-!
Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-margin

Exact scaled-integer scores model dyadic inputs. This theorem proves separation,
not Python conversion, sorting, CUDA execution or future-round error bounds.
-/
namespace AssemblyIR

/-- A gap strictly greater than twice the error preserves a selected/outsider pair. -/
theorem winner_pair_separated (selected outsider actualSelected actualOutsider error : Int)
    (gap : outsider + 2 * error < selected)
    (lower : selected - error <= actualSelected)
    (upper : actualOutsider <= outsider + error) :
    actualOutsider < actualSelected := by omega

/-- Apply the pair theorem to every selected and unselected index. -/
theorem winner_set_separated {Index : Type} (chosen : Index -> Prop)
    (reference actual : Index -> Int) (error : Int)
    (lower : forall i, reference i - error <= actual i)
    (upper : forall i, actual i <= reference i + error)
    (gap : forall i j, chosen i -> Not (chosen j) ->
      reference j + 2 * error < reference i) :
    forall i j, chosen i -> Not (chosen j) -> actual j < actual i := by
  intro i j hi hj
  exact winner_pair_separated _ _ _ _ error (gap i j hi hj) (lower i) (upper j)

/-- Strictness is necessary: at equality an outsider can tie a selected score. -/
example : (0 : Int) + 2 * 1 = 2 /\ Not ((1 : Int) < 1) := by decide

end AssemblyIR
