import os
import shutil
from glob import glob
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


train_r = 0.7
val_r = 0.1
test_r = 0.2
seed = 224

db_path_root = "/home/ahanaf/Datasets/Eye_Disease_Image_Dataset"
csv_name = "db_split.csv"

image_files = sorted(glob(f'{db_path_root}/*/*'))
image_files = ["/".join(p.split("/")[-2:]) for p in image_files]

labels = [x.split('/')[0] for x in image_files]

df = pd.DataFrame({'image_path': image_files, 'label': labels})

le = LabelEncoder()
label_encoded = le.fit_transform(df['label'])
df['label_encoded'] = label_encoded

class_map = {i: le.classes_[i] for i in range(len(le.classes_))}
with open("data_split/class_map.json", "w") as f:
    json.dump(class_map, f, indent=2)

images_train, images_test, labels_train, labels_test, le_train, le_test = train_test_split(
    image_files, labels, label_encoded,
    train_size=train_r,
    random_state=seed,
    stratify=labels
)

images_test, images_val, labels_test, labels_val, le_test, le_val = train_test_split(
    images_test, labels_test, le_test,
    train_size=0.5,
    random_state=seed,
    stratify=labels_test
)

df_train = pd.DataFrame({'image_path': images_train, 'label': labels_train, 'labels_encoded': le_train, 'split': 'train'})
df_val = pd.DataFrame({'image_path': images_val, 'label': labels_val, 'labels_encoded': le_val, 'split': 'val'})
df_test = pd.DataFrame({'image_path': images_test, 'label': labels_test, 'labels_encoded': le_test, 'split': 'test'})

df_final = pd.concat([df_train, df_val, df_test], axis=0)

local_csv_path = os.path.join("data_split", csv_name)
df_final.to_csv(local_csv_path, index=False)

os.makedirs(db_path_root, exist_ok=True)
shutil.copy2(local_csv_path, os.path.join(db_path_root, csv_name))
shutil.copy2('data_split/class_map.json', os.path.join(db_path_root, 'class_map.json'))

print(len(df_final))
print(df_final['split'].value_counts())