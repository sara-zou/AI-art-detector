import shutil
import os
import random
import uuid

ai_source = "ai_art_data"

train_dest = "dataset/train/ai"
val_dest   = "dataset/val/ai"
test_dest  = "dataset/test/ai"

human_train = "dataset/train/human"
human_val   = "dataset/val/human"
human_test  = "dataset/test/human"

def count_images(folder):
    return len([f for f in os.listdir(folder) 
                if f.lower().endswith((".jpg", ".png", ".jpeg"))])

def copy_images(img_list, dest):
    os.makedirs(dest, exist_ok=True)
    for i, img in enumerate(img_list):
        ext = os.path.splitext(img)[1]
        new_name = f"{uuid.uuid4().hex}{ext}"
        shutil.copy2(img, os.path.join(dest, new_name))

train_human = count_images(human_train)
val_human   = count_images(human_val)
test_human  = count_images(human_test)

print(f"Human images -> Train: {train_human}, Val: {val_human}, Test: {test_human}")

ai_images = []
for root, _, files in os.walk(ai_source):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            ai_images.append(os.path.join(root, file))

print("Total AI images found:", len(ai_images))

total_needed = train_human + val_human + test_human
if len(ai_images) < total_needed:
    raise ValueError(f"Not enough AI images! Needed {total_needed}, found {len(ai_images)}")

random.shuffle(ai_images)

train_imgs = ai_images[:train_human]
val_imgs   = ai_images[train_human:train_human + val_human]
test_imgs  = ai_images[train_human + val_human:
                       train_human + val_human + test_human]

print(f"Using AI images -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("Copying train...")
copy_images(train_imgs, train_dest)

print("Copying val...")
copy_images(val_imgs, val_dest)

print("Copying test...")
copy_images(test_imgs, test_dest)

print("Done!")

print("Final counts:")
print("Train AI:", count_images(train_dest), "Train Human:", train_human)
print("Val AI:", count_images(val_dest), "Val Human:", val_human)
print("Test AI:", count_images(test_dest), "Test Human:", test_human)