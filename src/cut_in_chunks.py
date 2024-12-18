import pandas as pd
import dvc.api

if __name__ == '__main__':
    params = dvc.api.params_show()
    chunk_size = params['chunk_size']
    chunk_overlap = params['chunk_overlap']

    annotated_data = pd.read_csv('./data/annotation_per_row_data.csv')
    
    # Create chunks using list comprehension instead of iterative append
    chunks = [
        row.to_dict() | {'start': chunk_start / 1000, 'end': (chunk_start + chunk_size) / 1000}
        for _, row in annotated_data.iterrows()
        for chunk_start in range(
            int(row['start'] * 1000),  # convert to ms
            int(row['end'] * 1000) - chunk_size,
            chunk_overlap
        )
    ]
    
    # Create DataFrame from list of dictionaries
    result = pd.DataFrame(chunks)
    result.to_csv('./data/chunked_data.csv', index=False)
