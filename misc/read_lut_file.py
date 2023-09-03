import json


def read_lut_file(file_path):
    label_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Ignore comment lines
            if line.startswith('#'):
                continue

            # Split line into components
            components = line.strip().split()

            # Check if line has enough components
            if len(components) < 5:
                continue

            # Extract label number and region name
            label_number = int(components[0])
            region_name = components[1]

            # Save label number and region name to dictionary
            label_dict[label_number] = region_name

    return label_dict

def save_label_dict_to_file(label_dict, output_file):
    with open(output_file, 'w') as f:
        json.dump(label_dict, f, indent=4)


lut_file_path = r'/usr/local/freesurfer/FreeSurferColorLUT.txt'
output_file_path = "/mnt/data/DCM/tmp/aparc_atlas/label_dict.json"


label_dict = read_lut_file(lut_file_path)

# Save label_dict to a JSON file
save_label_dict_to_file(label_dict, output_file_path)




