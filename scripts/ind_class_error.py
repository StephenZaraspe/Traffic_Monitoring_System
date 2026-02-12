import os

# Your specific path
base_path = r'C:\Users\ADMIN\Downloads\traffic_monitoring_system\datasets\roboflow_ph_vehicles'
folders = ['train/labels', 'valid/labels', 'test/labels']

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        continue
        
    print(f"Processing {folder}...")
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            changes_made = False
            
            for line in lines:
                parts = line.split()
                if len(parts) > 0 and parts[0] == '6':
                    parts[0] = '5'  # Remap to 'truck'
                    new_lines.append(" ".join(parts) + "\n")
                    changes_made = True
                else:
                    new_lines.append(line)
            
            if changes_made:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)

print("Done! All Class 6 labels have been remapped to Class 5 (truck).")