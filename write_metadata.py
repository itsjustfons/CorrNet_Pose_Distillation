import os
import pickle
import pandas as pd
from tqdm import tqdm
import csv

# --- Paths (update if different on your machine) ---
OPENASL_TSV = "/data/group1/z40575r/GloFE/openasl-v1.0.tsv"
POSE_DIR = "/data/group1/z40575r/GloFE/tools/openasl_mmpose2/openasl_mmpose/"
OUTPUT_TSV = "pose_lengths.tsv"

def main():
    # Load original OpenASL TSV with split info
    df = pd.read_csv(OPENASL_TSV, sep="\t")

    # Prepare output file
    with open(OUTPUT_TSV, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["vid", "length", "split"])  # header

        # Iterate through dataset rows
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vid = row["vid"]
            split = row["split"]

            file_path = os.path.join(POSE_DIR, f"{vid}.pkl").replace(":", "-")

            # Try reading the .pkl
            try:
                with open(file_path, "rb") as f2:
                    pose_keypoints = pickle.load(f2)
                length = pose_keypoints.shape[0]
                writer.writerow([vid, length, split])
            except Exception as e:
                # Log problematic files but skip them
                print(f"Error {vid}: {e}")

    print(f"\n✅ Metadata written to: {OUTPUT_TSV}")

if __name__ == "__main__":
    main()
