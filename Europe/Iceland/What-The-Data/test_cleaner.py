import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FOLDER = Path(__file__).resolve().parent


def run_cleaner(export):
    with tempfile.TemporaryDirectory() as directory:
        working = Path(directory)
        (working / 'test-input.csv').write_bytes(export)
        completed = subprocess.run(
            [sys.executable, str(FOLDER / 'main.py')],
            cwd=working, capture_output=True, text=True, timeout=10,
        )
        if completed.returncode:
            raise AssertionError(completed.stderr)
        return (working / 'test-output.csv').read_text(encoding='utf-8')


class CleanerTests(unittest.TestCase):
    def test_practice_matches_supplied_answer(self):
        cleaned = run_cleaner((FOLDER / 'data' / 'train-input.csv').read_bytes())
        expected = (FOLDER / 'data' / 'train-output.csv').read_text(encoding='utf-8')
        self.assertEqual(cleaned, expected)

    def test_unseen_stations_repeated_encoding_and_cross_unit_duplicates(self):
        stations = ['Þingvellir', 'Reykjahlíð', 'Húsavík', 'Grjótagjá', 'Öxarfjörður']
        latin_broken = stations[0].encode('utf-8').decode('latin-1')
        windows_broken = stations[1].encode('utf-8').decode('cp1252')
        twice_broken = stations[2].encode('utf-8').decode('cp1252')
        twice_broken = twice_broken.encode('utf-8').decode('cp1252')
        export = io.StringIO()
        export.write('\ufeff' + '# legacy relay\n\n' * 20)
        writer = csv.writer(export, delimiter=';', lineterminator='\n')
        writer.writerow(['reading_id', 'station_name', 'temperature_raw', 'unit', 'reading_date'])
        rows = [
            [10, stations[3], '-40,0', 'F', '2026-11-14'],
            [2, latin_broken, '212,0', 'F', '2026-11-14'],
            [1, windows_broken, '32.0', 'F', '2026-11-14'],
            [3, twice_broken, '54,6', 'C', '2026-11-14'],
            [4, stations[4], '61.3', 'C', '2026-11-14'],
            [2, stations[0], '100.0', 'C', '2026-11-14'],
            [3, stations[2], '54.6', 'C', '2026-11-14'],
        ]
        writer.writerows(rows + rows[::-1] * 8)
        cleaned = run_cleaner(export.getvalue().encode('utf-8'))
        expected = (
            'reading_id,station_name,temperature_c,reading_date\n'
            '1,Reykjahlíð,0.0,2026-11-14\n'
            '2,Þingvellir,100.0,2026-11-14\n'
            '3,Húsavík,54.6,2026-11-14\n'
            '4,Öxarfjörður,61.3,2026-11-14\n'
            '10,Grjótagjá,-40.0,2026-11-14\n'
        )
        self.assertEqual(cleaned, expected)

    def test_header_without_preamble_and_all_fahrenheit(self):
        export = (
            'reading_id;station_name;temperature_raw;unit;reading_date\r\n'
            '2;Ásbyrgi;168,8;F;2026-01-01\r\n'
            '1;Laugarvatn;173.5;F;2026-01-02\r\n'
        )
        self.assertEqual(
            run_cleaner(export.encode('utf-8')),
            'reading_id,station_name,temperature_c,reading_date\n'
            '1,Laugarvatn,78.6,2026-01-02\n'
            '2,Ásbyrgi,76.0,2026-01-01\n',
        )

    def test_windows_encoding_of_uppercase_thorn(self):
        station = 'Þórsmörk'.encode('utf-8').decode('cp1252')
        export = (
            'reading_id;station_name;temperature_raw;unit;reading_date\n'
            f'1;{station};54,6;C;2026-11-14\n'
        )
        self.assertEqual(
            run_cleaner(export.encode('utf-8')),
            'reading_id,station_name,temperature_c,reading_date\n'
            '1,Þórsmörk,54.6,2026-11-14\n',
        )
