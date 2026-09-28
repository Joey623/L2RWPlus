import contextlib
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from l2rwplus.cli import evaluate


class EvaluateCLITest(unittest.TestCase):
    def test_trial_selects_corresponding_release_checkpoint(self):
        for trial in range(1, 11):
            args = evaluate.parse_args(['--dataset', 'regdb', '--trial', str(trial)])
            self.assertEqual(args.checkpoints, [(trial, Path(f'checkpoints/regdb/trial_{trial:02d}.pth'))])
        args = evaluate.parse_args(['--dataset', 'sysu'])
        self.assertEqual(args.checkpoints[0][1], Path('checkpoints/sysu_uci.pth'))

    def test_invalid_combinations(self):
        invalid = [
            ['--dataset', 'sysu', '--all-trials'],
            ['--dataset', 'regdb', '--all-trials', '--trial', '1'],
            ['--dataset', 'regdb', '--all-trials', '--checkpoint', 'global.pth'],
            ['--dataset', 'regdb', '--checkpoint', '{unknown}.pth'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.parse_args(argv)

    def test_all_trials_uses_each_model_and_averages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for trial in range(1, 11):
                (root / f'trial_{trial:02d}.pth').touch()
            output = root / 'summary.json'
            def fake(args, path, trial):
                self.assertEqual(path, root / f'trial_{trial:02d}.pth')
                return dict(trial=trial, r1=trial, mAP=trial * 2, mINP=trial * 3)
            with patch.object(evaluate, 'evaluate_checkpoint', side_effect=fake) as call, contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(['--dataset', 'regdb', '--all-trials', '--checkpoint', str(root / 'trial_{trial:02d}.pth'), '--output', str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(call.call_count, 10)
            self.assertEqual([r['trial'] for r in result['trials']], list(range(1, 11)))
            self.assertEqual(result['mean'], dict(r1=5.5, mAP=11, mINP=16.5))
            self.assertAlmostEqual(result['sample_std']['r1'], statistics.stdev(range(1, 11)))

    def test_missing_trial_prevents_partial_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for trial in range(1, 10):
                (root / f'trial_{trial:02d}.pth').touch()
            with patch.object(evaluate, 'evaluate_checkpoint') as call, self.assertRaises(SystemExit):
                evaluate.main(['--dataset', 'regdb', '--all-trials', '--checkpoint', str(root / 'trial_{trial:02d}.pth')])
            call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
