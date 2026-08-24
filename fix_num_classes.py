import json
with open('dwopp-for-ip102.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '--num_classes' in source:
            new_source = source.replace("--num_classes', '16',", "--num_classes', '6',")
            cell['source'] = new_source.split('\n')
            print(f'Updated cell {i}')
            break

with open('dwopp-for-ip102.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
print('Done')