import csv
from decimal import Decimal


def restore_station(station):
    while True:
        for encoding in ('latin-1', 'cp1252'):
            try:
                repaired = station.encode(encoding).decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if len(repaired) < len(station):
                station = repaired
                break
        else:
            return station


readings = {}
with open('test-input.csv', encoding='utf-8-sig', newline='') as source:
    lines = (line for line in source if line.strip() and not line.lstrip().startswith('#'))
    for reading in csv.DictReader(lines, delimiter=';'):
        reading_id = int(reading['reading_id'])
        station = restore_station(reading['station_name'])
        temperature = Decimal(reading['temperature_raw'].strip().replace(',', '.'))
        unit = reading['unit'].strip().upper()
        if unit == 'F':
            temperature = (temperature - Decimal('32')) * Decimal('5') / Decimal('9')
        elif unit != 'C':
            raise ValueError(f'Unknown temperature unit: {unit}')
        readings.setdefault(reading_id, (station, f'{temperature:.1f}', reading['reading_date']))

with open('test-output.csv', 'w', encoding='utf-8', newline='') as target:
    writer = csv.writer(target, quoting=csv.QUOTE_NONE, lineterminator='\r\n')
    writer.writerow(['reading_id', 'station_name', 'temperature_c', 'reading_date'])
    for reading_id in sorted(readings):
        writer.writerow([reading_id, *readings[reading_id]])
