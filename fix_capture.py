import json
with open('dwopp-for-ip102.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'capture_output=False' in source:
            new_source = source.replace(
                "result = subprocess.run(cmd, capture_output=False)",
                "result = subprocess.run(cmd, capture_output=True, text=True)\n    if result.returncode != 0:\n        print('STDOUT:', result.stdout[-2000:])\n        print('STDERR:', result.stderr[-2000:])"
            )
            cell['source'] = new_source.split('\n')
            print(f'Updated cell {i}')
            break

with open('dwopp-for-ip102.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
print('Done')