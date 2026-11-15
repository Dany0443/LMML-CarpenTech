from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = (ROOT / 'test-output.txt').read_text(encoding='utf-8').splitlines()
assert len(lines) == 28, 'The submission must contain exactly 28 lines'
assert len(set(lines)) == 28, 'Duplicate filenames are not allowed'
for line in lines:
    relative = Path(line)
    assert relative.parent == Path('images'), 'Use images/filename.png on each line'
    assert relative.suffix == '.png'
    assert (ROOT / relative).is_file(), f'Missing image: {line}'
print('PASS: 28 unique, existing image paths in the required format')
