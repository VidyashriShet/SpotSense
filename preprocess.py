import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
dataset_path = "spotsense_temples_dataset"
augmented_path = "spotsense_temples_preprocessed_dataset"
os.makedirs(augmented_path, exist_ok=True)

# Function to dynamically choose the best edge detection method
def dynamic_edge_detection(image):
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges_canny = cv2.Canny(gray_image, 100, 200)
    
    # Laplacian edge detection
    edges_laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    edges_laplacian = cv2.convertScaleAbs(edges_laplacian)
    
    # Compute the total intensity for both edge images
    intensity_canny = np.sum(edges_canny)
    intensity_laplacian = np.sum(edges_laplacian)
    
    # Select the edge detection result with higher intensity
    if intensity_canny >= intensity_laplacian:
        return edges_canny, "Canny"
    else:
        return edges_laplacian, "Laplacian"

# Image augmentation function
def preprocess_image(image):
    rows, cols, _ = image.shape
    augmented_images = []

    # Affine transformations
    affine_matrices = [
        # Rotation
        cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1),
        # Translation
        cv2.getAffineTransform(
            np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]]),
            np.float32([[50, 50], [cols - 100, 50], [50, rows - 100]])
        ),
        # Shearing
        cv2.getAffineTransform(
            np.float32([[50, 50], [200, 50], [50, 200]]),
            np.float32([[30, 60], [220, 20], [50, 250]])
        )
    ]

    # Apply each affine transformation
    for matrix in affine_matrices:
        transformed_image = cv2.warpAffine(image, matrix, (cols, rows))
        augmented_images.append(transformed_image)

    final_images = []
    for img in augmented_images:
        # Scaling and Resizing
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Histogram Equalization
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            img = cv2.equalizeHist(img)

        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

        # Apply dynamic edge detection
        edges, method = dynamic_edge_detection(img)
        final_images.append((img, edges, method))

    return final_images

# Display before and after preprocessing
def display_images(original, processed, edges, method):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title("Processed Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Edges ({method} Detection)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Process and display 5 sample images
count = 0
temple_folders = os.listdir(dataset_path)

with tqdm(total=len(temple_folders)) as pbar:
    for temple_folder in temple_folders:
        temple_path = os.path.join(dataset_path, temple_folder)
        save_path = os.path.join(augmented_path, temple_folder)
        os.makedirs(save_path, exist_ok=True)

        for img_file in os.listdir(temple_path):
            img_path = os.path.join(temple_path, img_file)
            image = cv2.imread(img_path)

            if image is None:
                continue  # Skip if image cannot be read

            # Convert all images to JPG format
            jpg_save_path = os.path.join(save_path, f"{img_file.split('.')[0]}.jpg")
            cv2.imwrite(jpg_save_path, image)

            # Apply preprocessing
            augmented_images = preprocess_image(image)

            # Save augmented images
            for idx, (processed_img, edges, method) in enumerate(augmented_images):
                cv2.imwrite(os.path.join(save_path, f"{img_file.split('.')[0]}_aug_{idx + 1}.jpg"), processed_img)
                cv2.imwrite(os.path.join(save_path, f"{img_file.split('.')[0]}_aug_{idx + 1}_edges_{method}.jpg"), edges)

            # Display first 5 sets of augmented images
            if count < 5:
                for processed_img, edges, method in augmented_images:
                    display_images(image, processed_img, edges, method)
                    count += 1
                    if count >= 5:
                        break

        pbar.update(1)

print("Image preprocessing complete!")