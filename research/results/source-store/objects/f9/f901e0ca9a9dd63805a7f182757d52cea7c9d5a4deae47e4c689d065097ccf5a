import Std.Tactic

/-!
Specification: neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention

The first executable Python attention operator is a snapshot readout. This
module proves its structural bounds over finite supports; it does not model
floating-point softmax, Brain mutation, or learned fibers.
-/
namespace AssemblyIR

structure AttentionSchedule where
  topK : Nat
  outputSize : Nat
  topK_pos : 0 < topK
  outputSize_pos : 0 < outputSize

def AttentionSchedule.select {α : Type} (schedule : AttentionSchedule)
    (keys : List α) : List α := keys.take schedule.topK

def AttentionSchedule.bound {α : Type} (schedule : AttentionSchedule)
    (values : List α) : List α := values.take schedule.outputSize

theorem AttentionSchedule.selected_length_le {α : Type}
    (schedule : AttentionSchedule) (keys : List α) :
    (schedule.select keys).length ≤ schedule.topK := by
  exact List.length_take_le schedule.topK keys

theorem AttentionSchedule.output_length_le {α : Type}
    (schedule : AttentionSchedule) (values : List α) :
    (schedule.bound values).length ≤ schedule.outputSize := by
  exact List.length_take_le schedule.outputSize values

/- Constructed controls: zero-sized limits cannot inhabit the schedule. -/
example : ¬ (∃ schedule : AttentionSchedule, schedule.topK = 0) := by
  rintro ⟨schedule, h⟩
  exact (Nat.ne_of_gt schedule.topK_pos) h

example : ¬ (∃ schedule : AttentionSchedule, schedule.outputSize = 0) := by
  rintro ⟨schedule, h⟩
  exact (Nat.ne_of_gt schedule.outputSize_pos) h

#print axioms AttentionSchedule.selected_length_le
#print axioms AttentionSchedule.output_length_le

end AssemblyIR
