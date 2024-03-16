import matplotlib.pyplot as plt
from PIL import Image
import os

data_dir_path = "/Data/Trans_Proj/data/street_view_images_raw/"

def plot_image_pairs(image_pairs, title=''):
    # Set up the figure size and subplots
    fig, axes = plt.subplots(len(image_pairs), 2, figsize=(10, 3 * len(image_pairs)))
    
    # If only one pair, axes dimensions are reduced
    if len(image_pairs) == 1:
        axes = [axes]
    
    for i, (similarity, query_filename, target_filename) in enumerate(image_pairs):
        # Construct the full path to the images
        query_image_path = os.path.join(data_dir_path, query_filename + '.jpg')
        target_image_path = os.path.join(data_dir_path, target_filename + '.jpg')
        
        # Load the images
        query_image = Image.open(query_image_path)
        target_image = Image.open(target_image_path)
        
        # Plot query image
        axes[i][0].imshow(query_image)
        axes[i][0].axis('off')  # Hide axes ticks
        
        # Plot target image
        axes[i][1].imshow(target_image)
        axes[i][1].axis('off')
        
        # Set title with similarity score
        axes[i][0].set_title(f'Query: {query_filename}')
        axes[i][1].set_title(f'Target: {target_filename} with Similarity: {similarity}')
        
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the big title
    plt.show()

## EigenPlaces
    
### Neigbhor Positive Examples

plot_image_pairs(highest_pairs_EP_neighbors, title='EigenPlaces Neighbors Highest Pairs')

### Neigbhor Negative Examples

plot_image_pairs(lowest_pairs_EP_neighbors, title='EigenPlaces Neighbors Lowest Pairs')

### Stranger Positive Examples

plot_image_pairs(highest_pairs_EP_strangers, title='EigenPlaces Strangers Highest Pairs')

### Stranger Negative Examples

plot_image_pairs(lowest_pairs_EP_strangers, title='EigenPlaces Strangers Lowest Pairs')

## MixVPR

### Neigbhor Positive Examples

plot_image_pairs(highest_pairs_MVPR_neighbors, title='MixVPR Neighbors Highest Pairs')

### Neigbhor Negative Examples

plot_image_pairs(lowest_pairs_MVPR_neighbors, title='MixVPR Neighbors Lowest Pairs')

### Stranger Positive Examples

plot_image_pairs(highest_pairs_MVPR_strangers, title='MixVPR Strangers Highest Pairs')

### Stranger Negative Examples

plot_image_pairs(lowest_pairs_MVPR_strangers, title='MixVPR Strangers Lowest Pairs')
