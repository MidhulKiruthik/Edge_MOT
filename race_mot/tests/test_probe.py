import unittest
from unittest.mock import patch
from types import SimpleNamespace

from race_mot.stream_probe import probe_source


class Capture:
    def __init__(self, count=3, fail_after=3):
        self.count = count
        self.fail_after = fail_after
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {1: 10, 2: 16, 3: 16, 4: self.count,
                5: self.position, 6: max(0, self.position - 1) * 100}.get(prop, 0)

    def read(self):
        if self.position >= self.fail_after:
            return False, None
        self.position += 1
        return True, SimpleNamespace(shape=(16, 16, 3))

    def release(self):
        self.released = True


CV = SimpleNamespace(CAP_PROP_FPS=1, CAP_PROP_FRAME_WIDTH=2,
                     CAP_PROP_FRAME_HEIGHT=3, CAP_PROP_FRAME_COUNT=4,
                     CAP_PROP_POS_FRAMES=5, CAP_PROP_POS_MSEC=6)


class ProbeTests(unittest.TestCase):
    def probe(self, capture):
        with patch.dict('sys.modules', {'cv2': CV}), patch(
            'race_mot.stream_probe._open_capture', return_value=(capture, 'fake')
        ):
            return probe_source('clip.avi', 1)

    def test_expected_file_end_is_not_read_error(self):
        capture = Capture()
        result = self.probe(capture)
        self.assertEqual(result['stop_reason'], 'expected_file_end')
        self.assertEqual(result['read_errors'], 0)
        self.assertEqual(result['frames_read'], 3)
        self.assertTrue(capture.released)
        self.assertEqual(result['decoder_observations']['frame_position']['last'], 3)
        self.assertEqual(result['decoder_observations']['timestamp_ms']['last'], 200)

    def test_early_file_failure_remains_error(self):
        result = self.probe(Capture(count=5))
        self.assertEqual(result['stop_reason'], 'read_failure')
        self.assertEqual(result['read_errors'], 1)

    def test_unknown_length_does_not_claim_eof(self):
        result = self.probe(Capture(count=0))
        self.assertEqual(result['stop_reason'], 'read_failure')

    def test_invalid_duration_rejected_before_decoder(self):
        for duration in (0, -1, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                probe_source('clip.avi', duration)

    def test_metadata_error_releases_capture(self):
        capture = Capture()
        capture.get = lambda prop: (_ for _ in ()).throw(ValueError("bad metadata"))
        with self.assertRaises(ValueError):
            self.probe(capture)
        self.assertTrue(capture.released)

    def test_network_read_failure_is_never_file_eof(self):
        capture = Capture()
        with patch.dict('sys.modules', {'cv2': CV}), patch(
            'race_mot.stream_probe._open_capture', return_value=(capture, 'fake')
        ):
            result = probe_source('http://camera/video', 1)
        self.assertEqual(result['stop_reason'], 'read_failure')
        self.assertEqual(result['read_errors'], 1)
