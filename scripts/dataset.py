import kagglehub

# Download latest version
path = kagglehub.dataset_download("dmitryshkadarevich/branch-prediction")

print("Path to dataset files:", path)