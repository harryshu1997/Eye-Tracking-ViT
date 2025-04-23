import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import h5py

class MPIFaceGazeDataset(Dataset):
    """
    Memory-efficient dataset class for MPIFaceGaze dataset.
    Instead of loading all data into memory, this class:
    1. Stores file paths and indices
    2. Loads data on-demand in __getitem__
    
    From debug output:
    - Root > Data (Group) > data: Shape (3000, 3, 448, 448), Type: float32
    - Root > Data (Group) > label: Shape (3000, 16), Type: float32
    """
    def __init__(self, root_dir, transform=None, use_all_labels=False, limit_per_participant=None):
        self.root_dir = root_dir
        self.transform = transform
        self.use_all_labels = use_all_labels
        self.training = False  # must be set externally
        
        # Instead of storing all data, just store file paths and indices
        self.samples = []  # Will contain tuples of (file_path, sample_index)
        
        # Track stats
        files_processed = 0
        total_samples = 0
        
        # Process each participant's data
        for filename in os.listdir(root_dir):
            if filename.endswith('.mat') and filename.startswith('p'):
                mat_path = os.path.join(root_dir, filename)
                participant_id = filename.split('.')[0]  # Get p00, p01, etc.
                
                try:
                    # Open file just to check structure and count samples
                    with h5py.File(mat_path, 'r') as f:
                        if 'Data' in f and 'data' in f['Data'] and 'label' in f['Data']:
                            data_ref = f['Data']['data']
                            num_samples = data_ref.shape[0]
                            
                            # Limit samples per participant if specified
                            if limit_per_participant:
                                samples_to_use = min(limit_per_participant, num_samples)
                            else:
                                samples_to_use = num_samples
                            
                            # Store file path and indices
                            for i in range(samples_to_use):
                                self.samples.append({
                                    'file_path': mat_path,
                                    'sample_idx': i,
                                    'participant_id': participant_id
                                })
                            
                            files_processed += 1
                            total_samples += samples_to_use
                            print(f"Indexed {participant_id}: {samples_to_use} samples")
                        else:
                            print(f"Warning: Expected data structure not found in {mat_path}")
                except Exception as e:
                    print(f"Error indexing {mat_path}: {e}")
        
        print(f"Dataset indexed: {files_processed} files, {total_samples} total samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        sample_idx = sample_info['sample_idx']
        
        # Load data on-demand
        try:
            with h5py.File(file_path, 'r') as f:
                # Access the Data group
                data_group = f['Data']
                
                # Get image data - shape (3, 448, 448)
                image_data = np.array(data_group['data'][sample_idx])
                
                # Get label data - shape (16,)
                label_data = np.array(data_group['label'][sample_idx])
                
                # Close file explicitly
                f.close()
                
            # Extract gaze direction (first 2 dimensions)
            gaze_direction = label_data[0:2]
            
            # Convert to PIL Image - need to transpose from (C, H, W) to (H, W, C)
            image_data = np.transpose(image_data, (1, 2, 0))
            
            # Ensure data is in correct range for PIL
            if image_data.dtype != np.uint8:
                if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                    if np.max(image_data) <= 1.0:
                        image_data = (image_data * 255).astype(np.uint8)
                    else:
                        image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            
            # Create PIL image
            image = Image.fromarray(image_data)
            
            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)
            
            # Apply label jittering if in training mode
            if self.training:
                jitter = 0.05  # jitter in normalized coordinates
                gaze_direction = gaze_direction.copy()  # Create a copy
                gaze_direction[0] += random.uniform(-jitter, jitter)
                gaze_direction[1] += random.uniform(-jitter, jitter)
                
                # Clamp values (if needed)
                gaze_direction[0] = min(max(gaze_direction[0], -1.0), 1.0)
                gaze_direction[1] = min(max(gaze_direction[1], -1.0), 1.0)
            
            # Return all labels or just gaze direction
            if self.use_all_labels:
                # Return all components
                head_pose = label_data[2:4] if len(label_data) > 3 else np.zeros(2)
                facial_landmarks = label_data[4:] if len(label_data) > 4 else np.zeros(12)
                
                gaze_tensor = torch.tensor(gaze_direction, dtype=torch.float32)
                head_pose_tensor = torch.tensor(head_pose, dtype=torch.float32)
                landmarks_tensor = torch.tensor(facial_landmarks, dtype=torch.float32)
                
                return image, {
                    'gaze': gaze_tensor,
                    'head_pose': head_pose_tensor,
                    'landmarks': landmarks_tensor
                }
            else:
                # Return only gaze direction
                gaze_tensor = torch.tensor(gaze_direction, dtype=torch.float32)
                return image, gaze_tensor
                
        except Exception as e:
            print(f"Error loading sample {idx} from {file_path}: {e}")
            # Return a dummy sample as fallback
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (224, 224), color='gray'))
            
            dummy_label = torch.zeros(2, dtype=torch.float32)
            return dummy_image, dummy_label