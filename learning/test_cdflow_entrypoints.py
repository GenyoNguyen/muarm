"""Integration checks for the paper CDFlow command routes."""
from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import learning.play as playback
import learning.train as training


class CDFlowEntrypointTests(unittest.TestCase):
    def test_training_names_the_field_checkpoint_cdflow(self) -> None:
        with TemporaryDirectory(prefix="muarm-cdflow-train-") as folder:
            dataset = Path(folder) / "buttons.npz"
            dataset.touch()
            argv = [
                "learning/train.py", "--robot", "panda", "--task", "cdflow_button",
                "--algo", "cdflow", "--dataset", str(dataset),
                "--save-dir", folder, "--cdflow-epochs", "1",
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.train_cdflow.train_cdflow"
            ) as fit:
                training.main()
            fit.assert_called_once()
            args, kwargs = fit.call_args
            self.assertEqual(args[0], str(dataset))
            self.assertEqual(args[1], str(Path(folder) / "panda_cdflow_button_cdflow.pt"))
            self.assertEqual(kwargs["epochs"], 1)

    def test_playback_uses_direct_contact_route_without_reach_env(self) -> None:
        with patch.object(playback, "make_env") as make_env, patch.object(
            playback, "_play_cdflow_contact"
        ) as project:
            playback.play(
                "panda", "cdflow_button", "cdflow", "/tmp/cdflow.pt", 2,
                render=False, max_steps=3, cdflow_mode="mc",
            )
        make_env.assert_not_called()
        project.assert_called_once()
        self.assertEqual(project.call_args.args[:4], ("/tmp/cdflow.pt", 2, 3, None))


if __name__ == "__main__":
    unittest.main()
