import json
with open('lỗi.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i in [6, 7]:
    cell = data['cells'][i]
    print(f'=== Cell {i} ===')
    print(f'Cell type: {cell["cell_type"]}')
    if 'outputs' in cell:
        for out in cell['outputs']:
            print(f'  Output type: {out.get("output_type")}')
            if 'text' in out:
                text = ''.join(out['text'])
                print(f'  Text: {text[:3000]}')
            if 'traceback' in out:
                print(f'  Traceback: {out["traceback"]}')
            if 'ename' in out:
                print(f'  Error name: {out["ename"]}')
                print(f'  Error value: {out["evalue"]}')
    print()