import pandas as pd
import yaml

OLD_DATA_PATH = "/data/local-files/?d=bell_detector_data/raw_data/"

if __name__ == "__main__":

    
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
        
    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")

    # Create chunks using list comprehension instead of iterative append
    chunks = [
        row.to_dict() | {"start": chunk_start, "end": (chunk_start + chunk_size)}
        for _, row in annotated_data.iterrows()
        for chunk_start in range(
            int(row["start"] * 1000),  # convert to ms
            int(row["end"] * 1000) - chunk_size,
            chunk_overlap,
        )
    ]

    # Create DataFrame from list of dictionaries
    result = pd.DataFrame(chunks)
    result.audio_path = result.audio_path.str.replace(OLD_DATA_PATH, "")
    result.to_csv("./data/chunked_data.csv", index=False)
