# Stego Degradation Analysis

This repository focuses on the analysis of steganographic image degradation through various transformations. The project allows the application of predefined transformations on stego images to evaluate their robustness under different conditions.

## Project Structure

- **core/**: Contains the transformation logic and utility functions.
- **configs/**: Includes the `transformations.yaml` file, which defines the transformations to be applied.
- **scripts/**: Houses scripts to execute transformation processes.
- **data/**: This folder should contain the dataset, including stego and transformed images.

## Prerequisites

- **Docker**: Ensure that Docker is installed on your system.
- **Dataset**: Download the `data` folder from OneDrive and place it inside the project directory.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jeffersonrodriguezc/stego_degradation_analysis.git
cd stego_degradation_analysis
```

### 2. Download and Place the Data

- Obtain the `data` folder from OneDrive.
- Place it inside the cloned repository.

### 3. Build the Docker Image

```bash
docker build -t stego-degradation-analysis:latest .
```

### 4. Run the Docker Container

To use Jupyter Lab and mount the current directory:

```bash
docker run -it --rm --gpus all -p 8888:8888 -v ${PWD}:/app stego-degradation-analysis:latest
```

This command:
- Starts an interactive session.
- Removes the container after exit.
- Maps port `8888` for Jupyter Lab access.
- Mounts the current directory to `/app` inside the container.

### 5. Apply Transformations

Open another terminal and enter the container:

```bash
docker exec -it <idcontainer> /bin/bash
```

Run the transformation script:

```bash
python scripts/apply_transformations.py
```

**Note:** The transformation settings, including input directory, must be modified inside the script before execution.

## Configuration File: `transformations.yaml` (it is already created)

This YAML file defines the transformation pipeline. Each transformation includes:

- **Name**: Identifier of the transformation.
- **Variations**: Different parameter sets for each transformation.
- **titles**: Titles for the variations.

Example structure:

```yaml
transformations:
  - name: resize
    variations:
      - scale: 0.999
      - scale: 0.975
  - name: gaussian_noise
    variations:
      - { mean: 0, sigma: 2 }
      - { mean: 0, sigma: 4 }
```

Ensure the transformations are defined in the desired order of application.

## Notes

- Ensure the `data` folder contains the necessary steganographic images.
- Modify the script `apply_transformations.py` to adjust paths and configurations before execution.
- This image contains tensorflow and pytorch for GPU support if it needs to be used.


