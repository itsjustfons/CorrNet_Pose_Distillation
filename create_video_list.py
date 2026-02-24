import csv

def create_vid_list(tsv_path, output_txt_path, file_prefix):
    valid_splits = {"valid", "test"}
    with open(tsv_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        valid_vids = [
            f"{file_prefix}/{row['vid']}.mp4"
            for row in reader if row['split'].strip() in valid_splits
        ]

    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        for vid in valid_vids:
            out_file.write(vid + '\n')

# Example usage:
tsv_path = '/data/group1/z40575r/GloFE/openasl-v1.0.tsv'
output_txt_path = './valid_test_vids.txt'
file_prefix = '/data/group1/z40575r/GloFE/cropped-vid'  # <-- Customize this
create_vid_list(tsv_path, output_txt_path, file_prefix)
