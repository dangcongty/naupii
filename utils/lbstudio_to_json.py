import json

json_path = 'datasets/project-140-at-2025-11-27-01-12-80d5f9b3.json'
with open(json_path, 'r') as f:
    data = json.load(f)


output_path = 'datasets/data.json'

outputs = {}

for dta in data:
    annos = dta['annotations']
    for anno in annos:
        for res in anno['result']:
            orig_h, orig_w = res['original_height'], res['original_width']
            x = (res['value']['x'] * orig_w)/100
            y = (res['value']['y'] * orig_h)/100
            if dta['file_upload'] not in outputs:
                outputs[dta['file_upload']] = []
            outputs[dta['file_upload']].append([x, y])

with open(output_path, 'w') as f:
    json.dump(outputs, f)