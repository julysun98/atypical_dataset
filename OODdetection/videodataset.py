import os
import decord
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 视频数据集类
class VideoDataset(Dataset):
    def __init__(self, video_dir, label_file, clip_len=16, frame_interval=1, num_clips=1, transform=None):
        """
        Args:
            video_dir (string): Path to the video files.
            label_file (string): Path to the label file.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            num_clips (int): Number of clips to sample from each video.在这里只能设置为1
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.transform = transform
        
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
            self.labels = {line.split()[0]: int(line.split()[1]) for line in lines}
        
        self.video_files = list(self.labels.keys())

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[video_file]
        
        # Read video frames using Decord
        video_path = os.path.join(self.video_dir, video_file)
        # print(video_path)
        # vr = decord.VideoReader(video_path)
        # total_frames = len(vr)
        # if (total_frames - self.clip_len * self.frame_interval) < 0:
        #     print("video_path",video_path,total_frames, (total_frames - self.clip_len * self.frame_interval))#
        # # try:
        # #     vr = decord.VideoReader(video_path)
        # #     total_frames = len(vr)
        # #     # 如果需要继续处理视频数据，可以在这里加入相关代码
        # # except decord._ffi.base.DECORDError as e:
        # #     print(f"Error loading video {video_path}: {e}")
        # #     return None
        
        # # Sample clips
        # clips = []
        # # for _ in range(self.num_clips):
        # start_idx = np.random.randint(0, (total_frames - self.clip_len * self.frame_interval))
        # indices = [start_idx + i * self.frame_interval for i in range(self.clip_len)]
        # frames = vr.get_batch(indices).asnumpy()
        # clips.append(frames)
        try:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            high = total_frames - self.clip_len * self.frame_interval

            if high > 0:
                start_idx = np.random.randint(0, high)
                indices = [start_idx + i * self.frame_interval for i in range(self.clip_len)]
                frames = vr.get_batch(indices).asnumpy()
            else:
                # Handle short videos
                indices = list(range(0, total_frames, self.frame_interval))
                frames = vr.get_batch(indices).asnumpy()
                
                # Repeat frames if the video is too short
                while frames.shape[0] < self.clip_len:
                    frames = np.concatenate((frames, frames[:self.clip_len - frames.shape[0]]), axis=0)

            if self.transform:
                processed_frames = [self.transform(frame) for frame in frames]
                # Stack frames into a tensor with shape (C, T, H, W)
                clips = torch.stack(processed_frames, dim=1)
            else:
                # If no transform is provided, just convert frames to a tensor
                clips = torch.tensor(frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

            sample = {'frames': clips, 'label': label}
            return sample
        
        except TypeError as e:
            print(f"TypeError occurred while processing {self.video_files[idx]}: {str(e)}")
            raise e  # 重新抛出异常以保持原有堆栈信息
        
        except decord.DECORDError as e:
            print(f"Decord error occurred while processing {video_path}: {str(e)}")
            return None

        except Exception as e:
            print(f"Exception occurred while processing {video_path}: {str(e)}")
            return None  # 处理异常，防止程序崩溃

class VideoDataset_combine(Dataset):
    def __init__(self, video_dir1,video_dir2, label_file, clip_len=16, frame_interval=1, num_clips=1, transform=None):
        """
        Args:
            video_dir (string): Path to the video files.
            label_file (string): Path to the label file.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            num_clips (int): Number of clips to sample from each video.在这里只能设置为1
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dir1 = video_dir1
        self.video_dir2 = video_dir2
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.transform = transform
        
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
            self.labels = {line.split()[0]: int(line.split()[1]) for line in lines}
        
        self.video_files = list(self.labels.keys())

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[video_file]
        
        # Read video frames using Decord
        if 'atypical' in video_file:
            video_path = os.path.join(self.video_dir2, video_file)
        else:
            video_path = os.path.join(self.video_dir1, video_file)
        # print(video_path)
        # vr = decord.VideoReader(video_path)
        # total_frames = len(vr)
        # if (total_frames - self.clip_len * self.frame_interval) < 0:
        #     print("video_path",video_path,total_frames, (total_frames - self.clip_len * self.frame_interval))#
        # # try:
        # #     vr = decord.VideoReader(video_path)
        # #     total_frames = len(vr)
        # #     # 如果需要继续处理视频数据，可以在这里加入相关代码
        # # except decord._ffi.base.DECORDError as e:
        # #     print(f"Error loading video {video_path}: {e}")
        # #     return None
        
        # # Sample clips
        # clips = []
        # # for _ in range(self.num_clips):
        # start_idx = np.random.randint(0, (total_frames - self.clip_len * self.frame_interval))
        # indices = [start_idx + i * self.frame_interval for i in range(self.clip_len)]
        # frames = vr.get_batch(indices).asnumpy()
        # clips.append(frames)
        try:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            high = total_frames - self.clip_len * self.frame_interval

            if high > 0:
                start_idx = np.random.randint(0, high)
                indices = [start_idx + i * self.frame_interval for i in range(self.clip_len)]
                frames = vr.get_batch(indices).asnumpy()
            else:
                # Handle short videos
                indices = list(range(0, total_frames, self.frame_interval))
                frames = vr.get_batch(indices).asnumpy()
                
                # Repeat frames if the video is too short
                while frames.shape[0] < self.clip_len:
                    frames = np.concatenate((frames, frames[:self.clip_len - frames.shape[0]]), axis=0)

            if self.transform:
                processed_frames = [self.transform(frame) for frame in frames]
                # Stack frames into a tensor with shape (C, T, H, W)
                clips = torch.stack(processed_frames, dim=1)
            else:
                # If no transform is provided, just convert frames to a tensor
                clips = torch.tensor(frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

            sample = {'frames': clips, 'label': label}
            return sample
        
        except TypeError as e:
            print(f"TypeError occurred while processing {self.video_files[idx]}: {str(e)}")
            raise e  # 重新抛出异常以保持原有堆栈信息
        
        except decord.DECORDError as e:
            print(f"Decord error occurred while processing {video_path}: {str(e)}")
            return None

        except Exception as e:
            print(f"Exception occurred while processing {video_path}: {str(e)}")
            return None  # 处理异常，防止程序崩溃

class GaussianNoiseDataset(Dataset):
    def __init__(self, num_samples, clip_len=16, frame_interval=1,transform=None):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.transform = transform

        # 假设每帧的大小为 (224, 224, 3)，可以根据实际情况调整
        self.frame_shape = (224, 224, 3)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成高斯噪声
        frames = np.random.normal(loc=0.5, scale=0.5, size=(self.clip_len, *self.frame_shape)).astype(np.float32)
        frames_clipped = np.clip(frames, 0, 1)
        clips = torch.tensor(frames_clipped).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        # processed_frames = []
        # for frame in frames:
        #     # 将每一帧从 NumPy 数组转换为 PIL 图像，以便应用 transform
        #     # frame = Image.fromarray((frame * 255).astype(np.uint8))  # 转换为 uint8 并创建 PIL 图像
        #     frame = Image.fromarray(((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8))

        #     # 如果定义了 transform，则对每一帧进行变换
        #     if self.transform:
        #         frame = self.transform(frame)
        #     else:
        #         # 如果没有定义 transform，转换为 Tensor
        #         frame = transforms.ToTensor()(frame)

        #     processed_frames.append(frame)

        # # 将处理后的帧堆叠，变为 (T, C, H, W)
        # clips = torch.stack(processed_frames)  # (T, C, H, W)
        # # 重新排列维度，变为 (C, T, H, W)
        # clips = clips.permute(1, 0, 2, 3)

        # 模拟标签，假设每个样本的标签为 0
        label = 1
        sample = {'frames': clips, 'label': label}
        return sample

class BernoulliNoiseDataset(Dataset):
    def __init__(self, num_samples, clip_len=16, frame_interval=1, p=0.5,transform=None):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            p (float): Probability of a pixel being 1. Default is 0.5 (Bernoulli distribution).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.p = p
        self.transform = transform
        # 假设每帧的大小为 (224, 224, 3)，可以根据实际情况调整
        self.frame_shape = (224, 224, 3)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成伯努利噪声，数据形状为 (clip_len, H, W, C)
        frames = np.random.binomial(1, self.p, size=(self.clip_len, *self.frame_shape)).astype(np.float32)

        # 将噪声数据从 (T, H, W, C) 转换为 (C, T, H, W)
        clips = torch.tensor(frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        # processed_frames = []
        # for frame in frames:
        #     # 将每一帧从 NumPy 数组转换为 PIL 图像，以便应用 transform
        #     frame = Image.fromarray((frame * 255).astype(np.uint8))  # 转换为 uint8 并创建 PIL 图像

        #     # 如果定义了 transform，则对每一帧进行变换
        #     if self.transform:
        #         frame = self.transform(frame)
        #     else:
        #         # 如果没有定义 transform，转换为 Tensor
        #         frame = transforms.ToTensor()(frame)

        #     processed_frames.append(frame)

        # # 将处理后的帧堆叠，变为 (T, C, H, W)
        # clips = torch.stack(processed_frames)  # (T, C, H, W)
        # # print(clips.shape)
        # # 重新排列维度，变为 (C, T, H, W)
        # clips = clips.permute(1, 0, 2, 3)

        # 模拟标签，假设每个样本的标签为 0
        label = 0
        sample = {'frames': clips, 'label': label}
        return sample
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
class DynamicHandwrittenDataset(Dataset):
    def __init__(self, num_samples, clip_len=16, frame_interval=1, transform=None, digits_per_clip=1):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            transform (callable, optional): Optional transform to be applied on a sample.
            digits_per_clip (int): Number of handwritten digits to appear in each clip.
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.transform = transform
        self.digits_per_clip = digits_per_clip

        # Load MNIST dataset (or other handwritten dataset)
        self.mnist = datasets.MNIST(root='/bask/projects/j/jiaoj-rep-learn/qiyue/code/outlier-exposure/data/MNIST', train=True, download=False)

        self.mnist_data = self.mnist.data.numpy()  # Shape: (N, 28, 28)
        self.mnist_labels = self.mnist.targets.numpy()  # Shape: (N,)
        self.frame_shape = (224, 224)  # Resize each frame to this shape
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Initialize empty list to hold frames of the clip
        frames = []
        
        # For each frame in the clip, randomly pick `digits_per_clip` digits
        for frame_idx in range(0, self.clip_len * self.frame_interval, self.frame_interval):
            frame = Image.new('L', self.frame_shape)  # Create a blank image (grayscale)
            
            for _ in range(self.digits_per_clip):
                # Randomly select a digit from MNIST
                random_idx = np.random.randint(len(self.mnist_data))
                digit_img = self.mnist_data[random_idx]  # Shape: (28, 28)
                
                # Resize digit and paste it on the frame
                digit_img = Image.fromarray(digit_img).resize((28, 28))
                position = (
                    np.random.randint(0, self.frame_shape[0] - 28),
                    np.random.randint(0, self.frame_shape[1] - 28)
                )
                frame.paste(digit_img, position)
            
            # Apply transform if any (e.g., normalization, augmentation)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = transforms.ToTensor()(frame)  # Convert to Tensor
            
            frames.append(frame)

        # Stack frames into a tensor of shape (T, C, H, W) -> (clip_len, 1, H, W)
        clips = torch.stack(frames)  # Shape: (T, 1, H, W)
        clips = clips.permute(1, 0, 2, 3)  # Re-arrange to (C, T, H, W)

        # Generate a random label for this sample
        label = np.random.randint(0, 10)  # Assume labels are digits (0-9)

        # Return a dictionary with the frames and label
        sample = {'frames': clips, 'label': label}
        return sample

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
import random

class DynamicShapesDataset(Dataset):
    def __init__(self, num_samples, clip_len=16, frame_interval=1, transform=None, shape_count=1):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            clip_len (int): Number of frames in each clip.
            frame_interval (int): Interval between frames in each clip.
            transform (callable, optional): Optional transform to be applied on a sample.
            shape_count (int): Number of shapes per frame (red sphere, blue cube, green pyramid).
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.transform = transform
        self.shape_count = shape_count
        self.frame_shape = (224, 224)  # Image size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Initialize empty list to hold frames of the clip
        frames = []
        
        # Random initial positions and velocities for shapes
        positions = [self._random_position() for _ in range(self.shape_count)]
        velocities = [self._random_velocity() for _ in range(self.shape_count)]

        # For each frame in the clip, update positions and render shapes
        for frame_idx in range(0, self.clip_len * self.frame_interval, self.frame_interval):
            frame = Image.new('RGB', self.frame_shape, color=(255, 255, 255))  # Create a blank image (white background)
            draw = ImageDraw.Draw(frame)

            # Update positions and draw shapes
            for i in range(self.shape_count):
                positions[i] = self._update_position(positions[i], velocities[i])

                # Randomly select which shape to draw
                shape_type = random.choice(['sphere', 'cube', 'pyramid'])
                if shape_type == 'sphere':
                    self._draw_sphere(draw, positions[i], color=(255, 0, 0))  # Red sphere
                elif shape_type == 'cube':
                    self._draw_cube(draw, positions[i], color=(0, 0, 255))  # Blue cube
                else:
                    self._draw_pyramid(draw, positions[i], color=(0, 255, 0))  # Green pyramid

            # Apply transform if any (e.g., normalization, augmentation)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = transforms.ToTensor()(frame)  # Convert to Tensor
            
            frames.append(frame)

        # Stack frames into a tensor of shape (T, C, H, W) -> (clip_len, 3, H, W)
        clips = torch.stack(frames)  # Shape: (T, 3, H, W)
        clips = clips.permute(1, 0, 2, 3)  # Re-arrange to (C, T, H, W)

        # Return a dictionary with the frames and label (dummy label, e.g., 0)
        sample = {'frames': clips, 'label': 0}
        return sample

    def _random_position(self):
        """Generate a random initial position for the shape."""
        x = random.randint(20, self.frame_shape[0] - 20)
        y = random.randint(20, self.frame_shape[1] - 20)
        return [x, y]

    def _random_velocity(self):
        """Generate a random velocity for the shape."""
        vx = random.randint(-5, 5)
        vy = random.randint(-5, 5)
        return [vx, vy]

    def _update_position(self, position, velocity):
        """Update position based on velocity, ensuring it stays within frame bounds."""
        new_x = position[0] + velocity[0]
        new_y = position[1] + velocity[1]

        # Bounce if hitting the boundaries
        if new_x < 0 or new_x >= self.frame_shape[0]:
            velocity[0] *= -1
        if new_y < 0 or new_y >= self.frame_shape[1]:
            velocity[1] *= -1

        return [new_x + velocity[0], new_y + velocity[1]]

    def _draw_sphere(self, draw, position, color):
        """Draw a red sphere (circle) at the specified position."""
        radius = 15
        x, y = position
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _draw_cube(self, draw, position, color):
        """Draw a blue cube (square) at the specified position."""
        size = 30
        x, y = position
        draw.rectangle([x - size // 2, y - size // 2, x + size // 2, y + size // 2], fill=color)

    def _draw_pyramid(self, draw, position, color):
        """Draw a green pyramid (triangle) at the specified position."""
        size = 30
        x, y = position
        draw.polygon([(x, y - size), (x - size // 2, y + size // 2), (x + size // 2, y + size // 2)], fill=color)


# # Example usage:
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
# ])

# # Create dataset
# dataset = DynamicShapesDataset(num_samples=1000, clip_len=16, transform=transform, shape_count=3)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# # Iterate through the data
# for batch in dataloader:
#     frames = batch['frames']
#     labels = batch['label']
#     print(frames.shape)  # e.g., (16, 3, 16, 224, 224)
#     print(labels)



        #     if high <= 0:
        #         print(f"Skipping video {video_path} with total_frames={total_frames} and calculated high={high}")
        #         # return None  # 或者返回其他合适的值
        #     if high > 0:
        #         start_idx = np.random.randint(0, high)
        #         indices = [start_idx + i * self.frame_interval for i in range(self.clip_len)]
        #         frames = vr.get_batch(indices).asnumpy()
        #         if self.transform:
        #             processed_frames = [self.transform(frame) for frame in frames]
        #             # Stack frames into a tensor with shape (C, T, H, W)
        #             clips = torch.stack(processed_frames, dim=1)
        #         else:
        #             # If no transform is provided, just convert frames to a tensor
        #             clips = torch.tensor(frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

        #         sample = {'frames': clips, 'label': label}
                
        #         return sample

        # except Exception as e:
        #     print(f"Exception occurred while processing {video_path}: {str(e)}")
        #     return None  # 处理异常，防止程序崩溃
        
    
            # print("frames[0].shape",frames.shape)
            # print(len(clips))
        # Apply transforms
        # if self.transform:
        #     clips = [torch.stack([self.transform(frame) for frame in frames]) for frames in clips]
        #     # print(clips[0].shape)
        #     clips = np.stack(clips[0], axis=0)
            # print(clips.shape)
            # clips = self.transform(clips)
        
        # print("clips[0].shape",len(clips),clips[0].shape)
        # Stack clips
        # clips = torch.cat(clips, dim=1).permute(1, 0, 2, 3) # (num_clips, T, C, H, W) -> (T, num_clips * C, H, W) -> (num_clips * T, C, H, W)

        #clips = torch.stack(clips).permute(0, 4, 1, 2, 3)  # (N, T, H, W, C) -> (N, C, T, H, W)
        # print("clips.shape",len(clips),clips[0].shape)

        # sample = {'frames': clips, 'label': label}

        # return sample

# import os
# import decord
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import random

# # 视频数据集类
# class VideoDataset(Dataset):
#     def __init__(self, video_dir, label_file, num_frames=8, transform=None):
#         """
#         Args:
#             video_dir (string): Path to the video files.
#             label_file (string): Path to the label file.
#             num_frames (int): Number of frames to sample from each video.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.video_dir = video_dir
#         self.num_frames = num_frames
#         self.transform = transform
        
#         # Read the label file
#         with open(label_file, 'r') as f:
#             lines = f.readlines()
#             self.labels = {line.split()[0]: int(line.split()[1]) for line in lines}
        
#         self.video_files = list(self.labels.keys())

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_file = self.video_files[idx]
#         label = self.labels[video_file]
        
#         # Read video frames using Decord
#         video_path = os.path.join(self.video_dir, video_file)
#         vr = decord.VideoReader(video_path)
#         total_frames = len(vr)
        
#         # Randomly sample frames
#         indices = sorted(random.sample(range(total_frames), min(self.num_frames, total_frames)))
#         frames = vr.get_batch(indices).asnumpy()
        
#         # Apply transforms
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]
#         print("frames[0].shape",frames[0].shape)
#         # Stack frames
#         frames = torch.stack(frames).permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

#         sample = {'frames': frames, 'label': label}

#         return sample


