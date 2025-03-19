import json
import os

# Path to the notebook
notebook_path = '/tmp/tmp_nh992kk_run_antmikinka_tf-training_issue_1_79cf58fa/LSTM Model.ipynb'

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Replace cell #4 (index 3) directly with our custom implementation
notebook['cells'][3]['source'] = [
    "# Import our custom MinMaxScaler implementation instead of sklearn's\n",
    "from custom_scaler import CustomMinMaxScaler\n",
    "\n",
    "# Use our custom scaler\n",
    "sc = CustomMinMaxScaler(feature_range=(0, 1))\n",
    "training_set_scaled = sc.fit_transform(training_set)"
]
print(f"Replaced MinMaxScaler import in cell 3")

# Find the cell that creates X_test (with indentation issue)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'dataset_total = pd.concat' in ''.join(cell['source']):
        # Fix the indentation
        fixed_source = []
        for line in cell['source']:
            if line.strip() == 'X_test.append(inputs[i-60:i, 0])':
                fixed_source.append('    ' + line.strip() + '\n')
            else:
                fixed_source.append(line)
        notebook['cells'][i]['source'] = fixed_source
        print(f"Fixed indentation in cell {i}")

# Write the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook saved to {notebook_path}")