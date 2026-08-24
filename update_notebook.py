import json
with open('dwopp-for-ip102.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'CL_train_DwoPP.py' in source and '--lr' in source:
            new_source = source.replace("--lr', '2e-4',", "--lr', '1e-4',")
            new_source = new_source.replace("--lr_decay_start_epoch', str(epochs // 2),", "--lr_decay_start_epoch', '0',")
            new_source = new_source.replace("--margin', '0.4',", "--margin', '0.2',")
            new_source = new_source.replace("--dmml_dist_metric', 'euclidean',", "--dmml_dist_metric', 'cosine',")
            new_source = new_source.replace("--distillation_dist_metric', 'euclidean',", "--distillation_dist_metric', 'cosine',")
            cell['source'] = new_source.split('\n')
            print(f'Updated cell {i}')
            break

with open('dwopp-for-ip102.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
print('Done')