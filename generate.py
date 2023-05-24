import os

# Path to the folder containing the images
folder_path = 'maps/'

# generate a dataset.csv file from the subfolders of the folder_path and image files in them
with open('dataset.csv', 'w') as f: 
    f.write('image_path,label\n')
    for subfolder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder)):
            for image in os.listdir(os.path.join(folder_path, subfolder)):
                if image.endswith('.jpg'):
                    f.write(os.path.join(folder_path, subfolder, image) + ',' + subfolder + '\n')


