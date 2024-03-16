
import glob
import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as tvf
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
from tqdm import tqdm


data_dir_path = "/Data/Trans_Proj/data/street_view_images_raw/"
repo_path = "/users/eleves-a/2022/haiyang.jiang/Trans_Proj/repo/MixVPR/"
model_path = repo_path + "models/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"

import sys
sys.path.append(repo_path)
from main import VPRModel

output_dir = "/users/eleves-a/2022/haiyang.jiang/Trans_Proj/repo/VPR_Proj/result/MixVPR/"
test_neighbors_path = '/users/eleves-a/2022/haiyang.jiang/Trans_Proj/data/neighbors/new_test_neighbors.csv'
test_strangers_path = '/users/eleves-a/2022/haiyang.jiang/Trans_Proj/data/neighbors/new_test_strangers.csv'

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# load model
def load_model(model_path, device=device):
    # Note that images must be resized to 320x320
    model = VPRModel(backbone_arch='resnet50',
                    layers_to_crop=[4],
                    agg_arch='MixVPR',
                    agg_config={'in_channels': 1024,
                                'in_h': 20,
                                'in_w': 20,
                                'out_channels': 1024,
                                'mix_depth': 4,
                                'mlp_ratio': 1,
                                'out_rows': 4},
                    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path} Successfully!")

    return model


model = load_model(model_path)

# add transforms
transform = tvf.Compose([
    tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
    tvf.ToTensor(),
    tvf.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
])


def load_and_transform_image(image_path, transform=transform, device="cpu"):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).to(device)  # Add batch dimension and move to device
    return transformed_image


class ImagePairDataset(Dataset):
    def __init__(self, csv_file, slice=[0, -1], test=False):
        self.pairs_frame = pd.read_csv(csv_file, sep=',', dtype=str)
        print(f"Totally {len(self.pairs_frame)} data to be processed.")

        if slice[1] == -1:
            self.pairs_frame = self.pairs_frame.iloc[slice[0]:]
        else:
            self.pairs_frame = self.pairs_frame.iloc[slice[0]:slice[1]]

        print(f"This round, totally {len(self.pairs_frame)} data to be processed.")

        if test:
            self.pairs_frame = self.pairs_frame.iloc[:100]

        # Initialize an empty list to store preloaded images and labels
        self.preloaded_images = []
        self.image_pairs = []

        error_counter = 0
        for index, row in self.pairs_frame.iterrows():
            query_img_name = str(row[0])
            target_img_name = str(row[1])
            
            # Attempt to load and transform each image pair
            try:
                query_image = load_and_transform_image(
                                os.path.join(data_dir_path, query_img_name + ".jpg"), 
                                transform)
                
                target_image = load_and_transform_image(
                                os.path.join(data_dir_path, target_img_name + ".jpg"), 
                                transform)

                # Store the preloaded and transformed images
                self.preloaded_images.append((query_image, target_image))
                self.image_pairs.append((query_img_name, target_img_name))
            except Exception as e:
                # This catches any exception, logging the error and skipping the problematic image pair
                print(f"Warning: Skipping image pair {query_img_name}, {target_img_name} due to an error: {e}")
                error_counter += 1

        print(f"-----Totally {error_counter} errors happened during loading images.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # Returns the preloaded and processed image tensors along with their file names
        return self.preloaded_images[idx], self.image_pairs[idx]


def process_batch(model, device, img_batch):
    query_images, target_images = img_batch
    query_images = query_images.to(device)
    target_images = target_images.to(device)

    model.eval()
    with torch.no_grad():
        query_embeddings = model(query_images)
        target_embeddings = model(target_images)

    # Calculate cosine similarity
    similarities = torch.nn.functional.cosine_similarity(query_embeddings, target_embeddings)
    return similarities.cpu().numpy()


def pipeline(dataset, output_result_name, batch_size=128, num_worker=8, verbose=False):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)   # NOTE: GPU

    all_similarities = []
    # Open a CSV file to save the similarities
    with open(output_result_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Query', 'Target', 'Similarity'])

        for images_batch, filename_batch in tqdm(dataloader, desc="Processing batches"):

            similarities = process_batch(model, device, images_batch)
            all_similarities.extend(similarities)

            if verbose:
                print("query_images.shape", images_batch[0].shape)
                print("target_images.shape", images_batch[1].shape)
                print("similarities.shape", similarities.shape)
            
            # Write each pair and its similarity to the CSV file
            for i in range(len(filename_batch[0])):
                try:
                    writer.writerow([(filename_batch[0][i]), (filename_batch[1][i]), similarities[i].item()]) # Convert similarity to a Python scalar with .item()
                except Exception as e:
                    # If an IndexError is encountered, skip to the next iteration of the loop
                    print("!Error problem appear")
                    continue

        print("All similarities have been saved to similarities.csv")

        return all_similarities


def get_max_existing_number(dir_path):
    # 获取已存在的文件名中的最大编号
    existing_files = os.listdir(dir_path)
    max_existing_number = 0
    for filename in existing_files:
        parts = filename[:-4].split('_')
        if parts[-1].isdigit():
            max_existing_number = max(max_existing_number, int(parts[-1]))

    print(max_existing_number)
    return max_existing_number


# number of total pairs
pairs_frame = pd.read_csv(test_strangers_path, sep=',', dtype=str)
total_pairs = len(pairs_frame)
del pairs_frame

chunk_size = 5000

# Calculate the number of chunks needed (using ceiling division to include any partial chunk at the end)
num_chunks = -(-total_pairs // chunk_size)  # Ceiling division

# How many chunk have been processed
start_chunk = 1

# Loop through each chunk
for i in range(start_chunk, num_chunks):
    start_index = i * chunk_size
    end_index = start_index + chunk_size
    
    # Adjust the end index for the last chunk if it goes beyond the total length
    end_index = min(end_index, total_pairs)
    
    # Now you have the start and end indices for the current chunk
    print(f'Chunk {i+1}: Start at {start_index}, End before {end_index}')

    # WORKING
    # neighbors
    # neighbors_dataset = ImagePairDataset(test_neighbors_path, test=False, slice=[start_index,end_index])
    
    # max_existing_number = get_max_existing_number(output_dir + "neighbors/")
    # test_neighbors_similarities = pipeline(neighbors_dataset, output_dir + "neighbors/" + "test_neighbors_" + str(max_existing_number+1) + ".csv")

    # strangers
    strangers_dataset = ImagePairDataset(test_strangers_path, test=False, slice=[start_index,end_index])

    max_existing_number = get_max_existing_number(output_dir + "strangers/")
    test_strangers_similarities = pipeline(strangers_dataset, output_dir + "strangers/" + "test_strangers_" + str(max_existing_number+1) + ".csv")
