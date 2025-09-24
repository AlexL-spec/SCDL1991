import os
import random
from PIL import Image

# file path and import
train_dir = "C:/Users/24601/Desktop/GenImage/BigGAN/imagenet_ai_0419_biggan/train/ai"
val_dir   = "C:/Users/24601/Desktop/GenImage/BigGAN/imagenet_ai_0419_biggan/val/ai"
output_train_dir = "C:/Users/24601/Desktop/GenImage/BigGAN/imagenet_ai_0419_biggan/train/aisub"
output_val_dir   = "C:/Users/24601/Desktop/GenImage/BigGAN/imagenet_ai_0419_biggan/val/aisub"


os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# data cleaning and resizing
def clean_dataset(file_list, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    cleaned = []
    for f in file_list:
        try:
            with Image.open(f) as img:
                img = img.convert("RGB")
                img = img.resize((224, 224), Image.BILINEAR)

                out_path = os.path.join(temp_dir, os.path.basename(f))
                img.save(out_path, format="JPEG")
                cleaned.append(out_path)
        except Exception as e:
            print(f"skip the damaged file: {f}, error: {e}")
    return cleaned


train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
val_files   = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]


temp_train_dir = "C:/Users/24601/Desktop/GenImage/BigGAN/temp_train_cleaned"
temp_val_dir   = "C:/Users/24601/Desktop/GenImage/BigGAN/temp_val_cleaned"

clean_train = clean_dataset(train_files, temp_train_dir)
clean_val   = clean_dataset(val_files, temp_val_dir)

# random sampling
final_train = random.sample(clean_train, min(24000, len(clean_train)))
final_val   = random.sample(clean_val,   min(6000, len(clean_val)))

# save
for f in final_train:
    os.replace(f, os.path.join(output_train_dir, os.path.basename(f)))
for f in final_val:
    os.replace(f, os.path.join(output_val_dir, os.path.basename(f)))

print(f"number of pics in the training set: {len(final_train)} ")
print(f"number of pics in the validation set: {len(final_val)} ")


