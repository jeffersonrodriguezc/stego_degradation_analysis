import os
import sys
import cv2
import ast
import json
import pandas as pd
import multiprocessing

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from core.metrics import Metrics
from core.transformations import Transformations
from core.utils import load_yaml_config, save_image

def process_image(args):
    """
    Applies all transformations in the pipeline to an image and saves the results.
    """
    image_path, pipeline, output_dir, csv_records, metadata = args
    try:
        image = cv2.imread(image_path) # BGR
        if image is None:
            print(f"Error reading image: {image_path}")
            csv_records.append([DATASET_NAME, image_path, "N/A", "N/A","N/A", 
                                "N/A", MODEL_NAME, "N/A", "N/A", "Error: Could not read image"])
            return
        
        transformer = Transformations()
        results = transformer.apply_pipeline(image, pipeline)
        
        person_folder = os.path.basename(os.path.dirname(image_path))
        
        for idx_r, (transformation_name, transformed_images) in enumerate(results.items()):
            for idx, transformed_image in enumerate(transformed_images):
                evaluator = Metrics(image, transformed_image)
                metrics = evaluator.compute_all()                
                variant_name = f"{transformation_name}_{idx+1}"
                transformation_dir = os.path.join(output_dir, transformation_name, variant_name)
                os.makedirs(os.path.join(transformation_dir, person_folder), exist_ok=True)
                
                image_name = os.path.basename(image_path).replace("_stego", "")
                output_image_path = os.path.join(transformation_dir, person_folder, image_name)
                parameters = pipeline[idx_r]["variations"][idx] if "variations" in pipeline[idx_r] else {}

                try:
                    cv2.imwrite(output_image_path, transformed_image)
                    csv_records.append([DATASET_NAME, image_path, output_image_path, transformation_name, 
                                        variant_name, parameters, MODEL_NAME, metadata, metrics, "Success"])
                    print(f"Processed: {output_image_path}")
                except Exception as e:
                    print(f"Error saving {output_image_path}: {e}")
                    csv_records.append([DATASET_NAME, image_path, output_image_path, transformation_name, 
                                        variant_name, parameters, MODEL_NAME, metadata, metrics, f"Error: {str(e)}"])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        csv_records.append([DATASET_NAME, image_path, "N/A","N/A", "N/A", "N/A", MODEL_NAME, "N/A", "N/A", f"Error: {str(e)}"])

def apply_transformations_parallel(root_dir, config_path, csv_output, csv_hiding_summary):
    """
    Applies all transformations in parallel using apply_pipeline.
    """
    transformations_pipeline = load_yaml_config(config_path, "transformations")
    
    input_dir = os.path.join(root_dir, "stego")
    output_base_dir = os.path.join(root_dir, "transformations")
    
    manager = multiprocessing.Manager()
    csv_records = manager.list()

    if os.path.exists(csv_output):
        existing_df = pd.read_csv(csv_output)
        existing_records = existing_df.values.tolist()
        csv_records.extend(existing_records)
    
    image_paths = []
    
    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.endswith("_stego.png"):
                    image_path = os.path.join(person_path, image_file)
                    row = csv_hiding_summary[csv_hiding_summary['output_image'] == image_path]
                    metadata = ast.literal_eval(row.iloc[0]['metadata'])
                    image_paths.append((image_path, transformations_pipeline, 
                                        output_base_dir, csv_records, metadata))
    
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.5))
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_image, image_paths)
    
    df = pd.DataFrame(list(csv_records), columns=["dataset", "input_image", "output_image","transformation", "variant", 
                                                  "parameters", "model","metadata", "metrics", "status"])
    df.to_csv(csv_output, index=False)
    print(f"CSV saved: {csv_output}")

# Example usage
DATASET_NAME = 'CFD_one_shot'
MODEL_NAME = 'steguz'
root_directory = f"/app/data/processed/{DATASET_NAME}/{DATASET_NAME}_{MODEL_NAME}" # Change this path to the directory containing the stego images
config_file = "/app/configs/transformations.yaml" # don't change this path
csv_file = f"/app/data/processed/{DATASET_NAME}/transformations_summary.csv" # don't change this path
csv_hiding_summary = pd.read_csv(f"/app/data/processed/{DATASET_NAME}/hiding_summary.csv")

# execute this script using: python scripts/apply_transformations.py from the root directory of the project
# Apply transformations in parallel
apply_transformations_parallel(root_directory, config_file, csv_file, csv_hiding_summary)
