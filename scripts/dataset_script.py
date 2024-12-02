import os
import pandas as pd

# Creates mappings of each row 
def map_nfps(filename):
    df = pd.read_csv(filename)
    df.index += 2 # Start from index 2 (corresponding to the row number in the csv file)
    nfp_mapping = df["nfp"].to_dict()
    return nfp_mapping

# Matches the image filenames to the corresponding nfp of that image
# adds the necessary info into data structures
def match_images_labels(image_folder, image_paths, labels):
    for i, filename in enumerate(os.listdir(image_folder)):
        
        # Get row number from filname 
        row_num = int(filename[5:].split("_")[0])

        # Use row number to get nfp 
        nfp = nfp_map[row_num]

        image_paths.append(os.path.join(image_folder, filename))
        labels.append(nfp)
        
    
filename = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/data/XGStels/XGStels.csv"
# First create mappings of nfp for each row
nfp_map = map_nfps(filename)

# Populate the data structures
image_folder = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/stel_images" 
image_paths = []
labels = []
match_images_labels(image_folder, image_paths, labels)

# Make the dataset for the model
df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})






# CODE TO TEST OUTPUT 
# import os
# import pandas as pd

# def map_nfps(filename):
#     df = pd.read_csv(filename)
#     df.index += 2  # Start from index 2 (corresponding to the row number in the csv file)
#     nfp_mapping = df["nfp"].to_dict()

#     return nfp_mapping

# filename = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/data/XGStels/XGStels.csv"
# # First create mappings of nfp for each row
# nfp_map = map_nfps(filename)

# image_folder = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/stel_images" 
# output_file = "/home/exouser/Public/output.txt"  # Path to the output file

# with open(output_file, "w") as f:  # Open the file in write mode
#     for i, filename in enumerate(sorted(os.listdir(image_folder))):  # Ensure files are sorted
#         # Get row number from filename
#         row_num = int(filename[5:].split("_")[0])

#         # Use row number to get nfp
#         nfp = nfp_map[row_num]

#         # Write to file
#         f.write(f"{i}: row_num = {row_num}, nfp = {nfp}\n")

# print(f"Output written to {output_file}")
