import os
import pandas as pd

def get_data(file_path):
    df_iris = pd.read_csv(file_path, header=None)
    return df_iris


# Main execution block
if __name__ == "__main__":
    
    # Input and output paths from SageMaker Processing
    input_path = "/opt/ml/processing/input"
    output_path = "/opt/ml/processing/output"
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Find the input file (assuming there's only one)
    input_files = os.listdir(input_path)
    if not input_files:
        raise ValueError("No input files found in the input directory")
    
    # input_file_path = os.path.join(input_path, input_files[0])
    input_file_path = os.path.join(input_path, 'iris_dataset.csv') 
    output_file_path = os.path.join(output_path, "preprocessed_iris.csv") 
    

     # Execute loading step
    print(f"Loading data from: {input_file_path}")
    processed_data = get_data(input_file_path)
    
    print(f"Saving processed data to: {output_file_path}")
    processed_data.to_csv(output_file_path, index=False, header=False)
    
    print("Processing complete!") 