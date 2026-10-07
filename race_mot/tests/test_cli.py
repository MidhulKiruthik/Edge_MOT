import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from race_mot.cli import main


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        output = io.StringIO()
        with patch('sys.argv', ['race-mot', *args]), redirect_stdout(output):
            status = main()
        return status, output.getvalue()

    def test_environment_input_does_not_print_secret(self):
        secret = 'rtsp://user:private-password@camera/stream'
        with patch.dict(os.environ, {'TEST_INPUT_URL': secret}), patch(
            'race_mot.cli.probe_source', return_value={'source': 'redacted'}
        ) as probe:
            status, output = self.run_cli(['probe', '--input-env', 'TEST_INPUT_URL'])
        self.assertEqual(status, 0)
        probe.assert_called_once_with(secret, 30.0)
        self.assertNotIn(secret, output)

    def test_missing_environment_input_does_not_open_camera(self):
        with patch.dict(os.environ, {}, clear=True), patch('race_mot.cli.probe_source') as probe:
            status, _ = self.run_cli(['probe', '--input-env', 'TEST_INPUT_URL'])
        self.assertEqual(status, 2)
        probe.assert_not_called()

    def test_existing_report_is_preserved_before_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_text('original evidence')
            with patch('race_mot.cli.probe_source') as probe:
                status, _ = self.run_cli(['probe', '--input', 'clip.avi', '--output', str(path)])
            self.assertEqual(status, 2)
            self.assertEqual(path.read_text(), 'original evidence')
            probe.assert_not_called()
