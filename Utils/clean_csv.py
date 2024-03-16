import os
import csv

# Path to your directory containing images
image_directory_path = "/Data/Trans_Proj/data/street_view_images_raw/"

# Path to your CSV file
csv_file_path = '/users/eleves-a/2022/haiyang.jiang/Trans_Proj/data/neighbors/new_test_strangers.csv'

# Path for the new filtered CSV file
filtered_csv_file_path = '/users/eleves-a/2022/haiyang.jiang/Trans_Proj/data/clean_neighbors/new_test_strangers.csv'

# Get a list of all filenames in the directory
valid_filenames = set(os.listdir(image_directory_path))

# Open the original CSV, read with headers, filter it, and write to a new CSV
with open(csv_file_path, mode='r', newline='') as csv_file, \
     open(filtered_csv_file_path, mode='w', newline='') as filtered_csv_file:
    csv_reader = csv.DictReader(csv_file)
    fieldnames = csv_reader.fieldnames
    csv_writer = csv.DictWriter(filtered_csv_file, fieldnames=fieldnames)
    
    # Write the header to the new CSV
    csv_writer.writeheader()

    counter = 0

    for row in csv_reader:
        # Append the ".jpg" extension to the filenames
        query_filename = row['point'] + ".jpg"
        neighbor_filename = row['stranger'] + ".jpg"

        # Check if both filenames exist in the directory
        if query_filename in valid_filenames and neighbor_filename in valid_filenames:
            # Write the row to the new CSV if both filenames are valid
            csv_writer.writerow(row)
        else:
            print("file is missing")
            counter += 1

print("Filtered CSV with headers has been created.")
print("Totally {} files are missing".format(counter))
