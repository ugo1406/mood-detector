import os

# Path to your model folder
model_dir = os.path.expanduser("~/.cache/huggingface/hub/mo-thecreator/vit-Facial-Expression-Recognition")

def get_folder_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

size_bytes = get_folder_size(model_dir)
size_mb = size_bytes / (1024 * 1024)
print(f"Model folder size: {size_mb:.2f} MB")
