#import matplotlib.pyplot as plt
#import numpy as np
#group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
#group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))   #creating the figure and subplots
#axes[0].boxplot(group_A)
#axes[0].set_title('Box Plot for Group A')
#axes[0].set_ylabel('Measurement Values')
#axes[1].boxplot(group_B)
#axes[1].set_title('Box Plot for Group B')
#axes[1].set_ylabel('Measurement Values')
#fig.suptitle('Box Plots for Group A and Group B')
#plt.show()










#import numpy as np
#import matplotlib.pyplot as plt
#from google.colab import files
#uploaded = files.upload()
#file_path = list(uploaded.keys())[0]
#with open(file_path, 'r') as file:
#    genome_sequence = file.read().strip()
#genome_list = list(genome_sequence)
#genome_length = len(genome_list)
#t = np.linspace(0, 4 * np.pi, genome_length)  # 4*pi gives about 2 turns
#x = np.cos(t)
#y = np.sin(t)
#z = np.linspace(0, 5, genome_length)  # z increases linearly to spread out the helix vertically
#coordinates = np.column_stack((x, y, z))
#color_map = {
#    'A': 'red',
#    'T': 'green',
#    'C': 'blue',
#    'G': 'purple'
#}
#colors = [color_map[molecule] for molecule in genome_list]
#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors, s=100)  # Increase marker size
#ax.set_title('3D Helix Structure of Genome Sequence')
#ax.set_xlabel('X (cos(t))')
#ax.set_ylabel('Y (sin(t))')
#ax.set_zlabel('Z (linear increase)')
#ax.set_xticks(np.arange(-1, 1.5, 0.5))
#ax.set_yticks(np.arange(-1, 1.5, 0.5))
#legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=molecule)
#                  for molecule, color in color_map.items()]
#ax.legend(handles=legend_handles, title='Molecules')
#plt.show()











#import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
#import requests
#from io import BytesIO
#google_drive_url = 'https://drive.google.com/file/d/1jCQIWyGXWlBrLldGPdqV5IhQW8TeBH7m/view?usp=sharing'
#file_id = google_drive_url.split('/d/')[1].split('/view?usp=sharing')[0]
#direct_download_url = f'https://drive.google.com/uc?export=view&id={file_id}'  # to download image
#response = requests.get(direct_download_url)     #download
#img = Image.open(BytesIO(response.content))
#img_array = np.array(img)
#plt.figure(figsize=(10, 6))
#plt.subplot(2, 2, 1)
#plt.title('Original Image')
#plt.imshow(img_array)
#plt.axis('off')
#rotated_img = np.rot90(img_array)
#flipped_img = np.fliplr(img_array)
#plt.subplot(2, 2, 2)
#plt.title('Rotated Image')
#plt.imshow(rotated_img)
#plt.axis('off')
#plt.subplot(2, 2, 3)
#plt.title('Flipped Image')
#plt.imshow(flipped_img)
#plt.axis('off')
#gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
#plt.subplot(2, 2, 4)
#plt.title('Grayscale Image')
#plt.imshow(gray_img, cmap='gray')
#plt.axis('off')
#plt.tight_layout()
#plt.show()






#from sklearn.datasets import load_iris
#import numpy as np
#import matplotlib.pyplot as plt
#iris = load_iris()    # Accessing the features (data) using NumPy array
#X = np.array(iris.data)  # Features (sepal length, sepal width, petal length, petal width)
#Y = np.array(iris.target)  # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)
#mean = np.mean(X, axis=0)
#median = np.median(X, axis=0)
#std_dev = np.std(X, axis=0)
#min_values = np.min(X, axis=0)
#max_values = np.max(X, axis=0)
#print("Mean:", mean)
#print("Median:", median)
#print("Standard Deviation:", std_dev)
#print("Minimum Values:", min_values)
#print("Maximum Values:", max_values)
#sepal_length_width = X[:, :2]
#plt.figure(figsize=(12, 8))
#plt.subplot(2, 2, 1)
#plt.scatter(sepal_length_width[:, 0], sepal_length_width[:, 1], c=Y, cmap='viridis', alpha=0.7)
#plt.title('Scatter Plot of Sepal Length vs Sepal Width')
#plt.xlabel('Sepal Length (cm)')
#plt.ylabel('Sepal Width (cm)')
#plt.colorbar(label='Species')
#plt.subplot(2, 2, 2)
#plt.hist(sepal_length_width[:, 0], bins=20, color='skyblue', edgecolor='black')
#plt.title('Histogram of Sepal Length')
#plt.xlabel('Sepal Length (cm)')
#plt.ylabel('Frequency')
#plt.subplot(2, 2, 3)
#for species in np.unique(Y):
#    species_data = X[Y == species]
#    plt.plot(species_data[:, 2], species_data[:, 3], label=iris.target_names[species])
#plt.title('Line Plot of Petal Length vs Petal Width')
#plt.xlabel('Petal Length (cm)')
#plt.ylabel('Petal Width (cm)')
#plt.legend(title='Species')
#plt.tight_layout()
#plt.show()

