import os
import unittest

from m3d.cli import (_render_leaderboard_md, _load_manifest, _submission_entries,
                     _load_runtimes)
from m3d.scorer import score_submission_set, rank_submissions, pareto_frontier

HARD_SUITE = "benchmarks_hard"
HARD_SUBS = "submissions/hard"


class TestLeaderboardRender(unittest.TestCase):
    def test_render_is_deterministic_and_lists_seed_entries(self):
        a = _render_leaderboard_md("submissions")
        b = _render_leaderboard_md("submissions")
        self.assertEqual(a, b)                     # deterministic
        self.assertIn("## hard", a)
        for name in ("negotiated", "negotiated_x2", "negotiated_fast"):
            self.assertIn(name, a)

    def test_committed_leaderboard_is_current(self):
        # the invariant CI enforces with `leaderboard-all --check`
        self.assertTrue(os.path.exists("LEADERBOARD.md"))
        on_disk = open("LEADERBOARD.md").read().strip()
        self.assertEqual(on_disk, _render_leaderboard_md("submissions").strip())


class TestSeedSubmissions(unittest.TestCase):
    def test_seed_hard_entries_score_as_expected(self):
        man = _load_manifest(HARD_SUITE)
        subs = {name: score_submission_set(man, HARD_SUITE, d, name, _load_runtimes(d))
                for name, d in _submission_entries(HARD_SUBS)}
        # two complete, one incomplete
        self.assertTrue(subs["negotiated"].complete)
        self.assertTrue(subs["negotiated_x2"].complete)
        self.assertFalse(subs["negotiated_fast"].complete)
        # baseline == the 'negotiated' reference, so its aggregate is exactly 1.0
        self.assertAlmostEqual(subs["negotiated"].aggregate, 1.0, places=6)
        # complete submissions rank ahead of the incomplete one
        ranked = rank_submissions(list(subs.values()))
        self.assertEqual(ranked[-1].name, "negotiated_fast")
        # Fast reference routers stay on the frontier because they spend less
        # time. cpp-route is the low-delay end of that frontier.
        self.assertEqual(set(pareto_frontier(list(subs.values()))),
                         {"cpp-route", "negotiated", "negotiated_x2"})


if __name__ == "__main__":
    unittest.main()
