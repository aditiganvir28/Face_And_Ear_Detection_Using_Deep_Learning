import os
import open3d as o3d
import numpy as np
import random

# Define augmentation functions

def normalize_point_cloud(pcd):
    """Normalize the point cloud to zero mean and unit variance."""
    points = np.asarray(pcd.points)
    # Center to zero mean
    centroid = points.mean(axis=0)
    points -= centroid
    # Scale to unit variance or unit sphere
    max_dist = np.linalg.norm(points, axis=1).max()
    points /= max_dist
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def rotate_point_cloud(pcd, axis='y', angle_range=(-30, 30)):
    """Rotate point cloud around a given axis."""
    angle = np.radians(random.uniform(*angle_range))
    rotation_matrix = np.eye(3)
    
    if axis == 'x':
        rotation_matrix = [[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]]
    elif axis == 'y':
        rotation_matrix = [[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]]
    elif axis == 'z':
        rotation_matrix = [[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]]
    
    return pcd.rotate(rotation_matrix)

def scale_point_cloud(pcd, scale_range=(0.8, 1.2)):
    """Randomly scale point cloud."""
    scale_factor = random.uniform(*scale_range)
    return pcd.scale(scale_factor, center=pcd.get_center())

def jitter_point_cloud(pcd, sigma=0.01, clip=0.05):
    """Add random noise to the point cloud."""
    points = np.asarray(pcd.points)
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    pcd.points = o3d.utility.Vector3dVector(points + noise)
    return pcd

def add_noise_to_point_cloud(pcd, noise_level=0.02):
    """Add uniform random noise to each point in the point cloud."""
    points = np.asarray(pcd.points)
    noise = noise_level * (np.random.rand(*points.shape) - 0.5)
    pcd.points = o3d.utility.Vector3dVector(points + noise)
    return pcd

def translate_point_cloud(pcd, translation_range=(0.1, 0.3)):
    """Translate point cloud randomly within a specified range."""
    translation = np.array([random.uniform(-translation_range[1], translation_range[1]) for _ in range(3)])
    pcd.translate(translation)
    return pcd

# Directory setup
input_dir = "dataset"
output_dir = "augmented_dataset"

# Create train and test directories
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Traverse dataset directories
for subject in os.listdir(input_dir):
    subject_path = os.path.join(input_dir, subject)
    if os.path.isdir(subject_path):
        sub3d_path = os.path.join(subject_path, "3D")
        if os.path.exists(sub3d_path):
            for file in os.listdir(sub3d_path):
                if file.endswith(".ply"):
                    file_path = os.path.join(sub3d_path, file)
                    
                    # Load point cloud
                    pcd = o3d.io.read_point_cloud(file_path)
                    # pcd = normalize_point_cloud(pcd)
                    
                    # Generate augmentations for training
                    for i in range(10):  # 6 augmentations for training
                        aug_pcd = rotate_point_cloud(pcd, axis='y')
                        aug_pcd = scale_point_cloud(aug_pcd)
                        aug_pcd = jitter_point_cloud(aug_pcd)
                        aug_pcd = add_noise_to_point_cloud(aug_pcd)
                        aug_pcd = translate_point_cloud(aug_pcd)
                        
                        # Construct output path for training
                        output_subject_dir = os.path.join(train_dir, subject, "3D")
                        if not os.path.exists(output_subject_dir):
                            os.makedirs(output_subject_dir)
                        
                        output_filename = f"{file[:-4]}_{i}.ply"
                        output_path = os.path.join(output_subject_dir, output_filename)
                        
                        # Save augmented point cloud
                        o3d.io.write_point_cloud(output_path, aug_pcd)
                    
                    # Generate augmentations for testing
                    for j in range(5):  # 2 augmentations for testing
                        aug_pcd = rotate_point_cloud(pcd, axis='y')
                        aug_pcd = scale_point_cloud(aug_pcd)
                        aug_pcd = jitter_point_cloud(aug_pcd)
                        aug_pcd = add_noise_to_point_cloud(aug_pcd)
                        aug_pcd = translate_point_cloud(aug_pcd)
                        
                        # Construct output path for testing
                        output_subject_dir = os.path.join(test_dir, subject, "3D")
                        if not os.path.exists(output_subject_dir):
                            os.makedirs(output_subject_dir)
                        
                        output_filename = f"{file[:-4]}_{j}.ply"
                        output_path = os.path.join(output_subject_dir, output_filename)
                        
                        # Save augmented point cloud
                        o3d.io.write_point_cloud(output_path, aug_pcd)

print("Data augmentation completed.")
