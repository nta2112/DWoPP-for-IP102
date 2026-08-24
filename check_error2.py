import json
with open('lỗi.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

cell = data['cells'][6]
if 'outputs' in cell:
    for out in cell['outputs']:
        print('Output type:', out.get('output_type'))
        for k, v in out.items():
            if k != 'text':
                print('  ', k, ':', v)