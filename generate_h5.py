import os
import numpy as np
import h5py
from plyfile import PlyData

# Constants
TRAIN_DIR = r'./augmented_dataset/train'
TEST_DIR = r'./augmented_dataset/test'
TRAIN_H5_PREFIX = r'./train_part_'  # Output train .h5 file prefix
TEST_H5_PREFIX = r'./test_part_'    # Output test .h5 file prefix
POINTS_PER_SAMPLE = 2048
SAMPLE_TYPES = ['earleft1', 'earleft2', 'earright1', 'earright2', 'face1', 'face2']

def load_ply(filename):
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return points

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    furthest_distance = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / furthest_distance
    return pc

def sample_points(pc, n_points=POINTS_PER_SAMPLE):
    if len(pc) >= n_points:
        indices = np.random.choice(len(pc), n_points, replace=False)
    else:
        indices = np.random.choice(len(pc), n_points, replace=True)
    return pc[indices]

def process_ply_files(dataset_dir, num_samples_per_file, num_files):
    data_collection = {}
    for part_idx in range(num_files):
        data_collection[part_idx] = ([], [])

    for person_folder in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        try:
            label = int(person_folder) - 1 
        except ValueError:
            continue  
        three_d_folder = os.path.join(person_path, '3D')
        if not os.path.exists(three_d_folder):
            continue
        
        samples = {sample_type: [] for sample_type in SAMPLE_TYPES}
        for filename in os.listdir(three_d_folder):
            if filename.endswith('.ply'):
                sample_type = '_'.join(filename.split('_')[:1])
                if sample_type in samples:
                    samples[sample_type].append(os.path.join(three_d_folder, filename))

        for sample_type, files in samples.items():
            if len(files) < num_samples_per_file * num_files:
                continue 

            np.random.shuffle(files)
            for part_idx in range(num_files):
                selected_files = files[part_idx * num_samples_per_file:(part_idx + 1) * num_samples_per_file]
                data_list, label_list = data_collection[part_idx]

                for file_path in selected_files:
                    pc = load_ply(file_path)
                    pc = normalize_point_cloud(pc)
                    pc = sample_points(pc, POINTS_PER_SAMPLE)

                    data_list.append(pc)
                    label_list.append([label])
    
    return data_collection

def shuffle_data_and_labels(data, labels):
    combined = list(zip(data, labels))
    np.random.shuffle(combined)
    shuffled_data, shuffled_labels = zip(*combined)
    return np.array(shuffled_data), np.array(shuffled_labels)

def save_to_h5(h5_filename, data, labels):
    data, labels = shuffle_data_and_labels(data, labels)
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype='float32')
        h5f.create_dataset('label', data=labels, compression='gzip', compression_opts=1, dtype='uint8')

# Main function to run the process
def main():
    train_data_collection = process_ply_files(TRAIN_DIR, num_samples_per_file=5, num_files=2)
    for part_idx, (data_list, label_list) in train_data_collection.items():
        train_h5_path = f"{TRAIN_H5_PREFIX}{part_idx + 1}.h5"
        save_to_h5(train_h5_path, np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.uint8))
        print(f"Saved {len(data_list)} train samples to {train_h5_path}")

    test_data_collection = process_ply_files(TEST_DIR, num_samples_per_file=5, num_files=1)
    for part_idx, (data_list, label_list) in test_data_collection.items():
        test_h5_path = f"{TEST_H5_PREFIX}{part_idx + 1}.h5"
        save_to_h5(test_h5_path, np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.uint8))
        print(f"Saved {len(data_list)} test samples to {test_h5_path}")

if __name__ == "__main__":
    main()
