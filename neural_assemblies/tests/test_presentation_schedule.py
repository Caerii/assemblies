"""The presentation-schedule instrument (PREREG_presentation_schedule.md).

The two schedules must differ in ORDER and in nothing else: same number of
presentations, same number per item. If that fails, the arms are not matched
and every comparison between them measures the count instead of the order.
"""
from __future__ import annotations

import unittest
from typing import Any

from research.experiments.presentation_schedule import (
    ARM_SPECS, ARMS, COMPARE_AT, EPISODE_ROUNDS, EPISODES, RULES, SCHEDULES,
    TOTAL_ROUNDS, SchedulePlan, elapse, visit_order,
)


class VisitOrder(unittest.TestCase):
    def test_the_schedules_are_matched_on_presentation_count(self):
        for items, visits in ((3, 4), (8, 16), (1, 2), (32, 1)):
            orders = {s: visit_order(s, items, visits) for s in SCHEDULES}
            for name, order in orders.items():
                self.assertEqual(len(order), items * visits, name)
                for i in range(items):
                    self.assertEqual(order.count(i), visits, (name, i))

    def test_massed_is_blocked_and_interleaved_is_round_robin(self):
        self.assertEqual(visit_order("massed", 3, 2), [0, 0, 1, 1, 2, 2])
        self.assertEqual(visit_order("interleaved", 3, 2), [0, 1, 2, 0, 1, 2])

    def test_the_schedules_differ_whenever_there_is_something_to_reorder(self):
        self.assertNotEqual(visit_order("massed", 3, 4), visit_order("interleaved", 3, 4))
        # ... and agree exactly when there is not, which is the degenerate case
        # a comparison between them cannot detect anything in.
        self.assertEqual(visit_order("massed", 1, 5), visit_order("interleaved", 1, 5))
        self.assertEqual(visit_order("massed", 5, 1), visit_order("interleaved", 5, 1))

    def test_an_unknown_schedule_is_refused(self):
        with self.assertRaises(ValueError):
            visit_order("spaced", 3, 4)

    def test_a_broken_schedule_would_fail_the_matching_check(self):
        # true negative: a schedule that drops the last visit of each item
        # still looks plausible, and the count check is what catches it.
        broken = [i for i in range(3) for _ in range(3)]
        self.assertNotEqual(len(broken), 3 * 4)
        self.assertTrue(any(broken.count(i) != 4 for i in range(3)))


class TimeHasNoHook(unittest.TestCase):
    def test_elapsing_time_touches_no_state(self):
        # SR-4's premise: there is no decay term, so an interval containing
        # nothing changes nothing. If this ever returns non-zero, idle spacing
        # has become a treatment and the registration's reasoning changes.
        for gap in (0, 1, 1000, 10**9):
            self.assertEqual(elapse(gap), 0)


class Plans(unittest.TestCase):
    def test_total_rounds_per_item_are_held_at_the_published_collapse_point(self):
        # every arm spends the same rounds per item; only their arrangement
        # differs, so a difference between arms cannot be a difference in
        # how much training each item received
        self.assertEqual(EPISODES * EPISODE_ROUNDS, TOTAL_ROUNDS)
        plan = SchedulePlan(schedule="massed", rule="control", strength=0.0,
                            checkpoints=(8,), visits=EPISODES,
                            episode_rounds=EPISODE_ROUNDS)
        self.assertEqual(plan.rounds_per_item, TOTAL_ROUNDS)
        # true negative: a plan that spends a different budget is refused,
        # because it would confound grouping with training amount
        with self.assertRaises(ValueError):
            SchedulePlan(schedule="massed", rule="control", strength=0.0,
                         checkpoints=(8,), visits=EPISODES + 1,
                         episode_rounds=EPISODE_ROUNDS)

    def test_every_arm_spends_the_same_rounds_per_item(self):
        # the arms may differ in GROUPING and ORDER and in nothing else, so a
        # difference between them is never a difference in training amount
        for name, (schedule, rule, episodes, rounds_each) in ARM_SPECS.items():
            self.assertIn(schedule, SCHEDULES, name)
            self.assertIn(rule, RULES, name)
            self.assertEqual(episodes * rounds_each, TOTAL_ROUNDS, name)

    def test_the_arms_cover_three_schedules_by_two_rules(self):
        self.assertEqual(len(ARMS), 6)
        for rule in RULES:
            for prefix in ("single", "massed", "interleaved"):
                self.assertIn(f"{prefix}-{rule}", ARMS)

    def test_the_single_arm_is_one_episode_and_the_others_are_split(self):
        for rule in RULES:
            self.assertEqual(ARM_SPECS[f"single-{rule}"][2], 1, rule)
            self.assertEqual(ARM_SPECS[f"massed-{rule}"][2], EPISODES, rule)
            self.assertEqual(ARM_SPECS[f"interleaved-{rule}"][2], EPISODES, rule)

    def test_the_comparison_checkpoints_are_fixed_numbers(self):
        # Amendment 1: never selected from the data. The version-1 rule chose
        # the one checkpoint where both arms were dead.
        self.assertEqual(len(COMPARE_AT), 2)
        for M in COMPARE_AT:
            self.assertIsInstance(M, int)

    def test_the_control_is_unrefracted_and_the_refracted_arm_is_not(self):
        self.assertEqual(RULES["control"], 0.0)
        self.assertGreater(RULES["refracted"], 0.0)

    def test_a_plan_refuses_what_it_cannot_run(self):
        def plan(*, schedule="massed", rule="control", strength=0.0,
                 checkpoints=(8, 16), visits=4, episode_rounds=4):
            return SchedulePlan(schedule=schedule, rule=rule, strength=strength,
                                checkpoints=checkpoints, visits=visits,
                                episode_rounds=episode_rounds)
        plan()                                        # the good one
        cases: tuple[tuple[str, dict[str, Any]], ...] = (
            ("unknown schedule", {"schedule": "spaced"}),
            ("unknown rule", {"rule": "hebbian"}),
            ("no visits", {"visits": 0}),
            ("no checkpoints", {"checkpoints": ()}),
            ("checkpoints out of order", {"checkpoints": (16, 8)}),
            ("repeated checkpoint", {"checkpoints": (8, 8)}),
            # an episode of one round inhibits, reads the stimulus alone and
            # stores nothing: a smoke run recalled at chance before this bar
            ("one-round episode", {"episode_rounds": 1}),
            ("zero-round episode", {"episode_rounds": 0}),
        )
        for label, kwargs in cases:
            with self.assertRaises(ValueError, msg=label):
                plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
