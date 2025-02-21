import argparse
import os
import shutil

import cv2
import cupy as cp
import numpy as np
import torch
from torchvision import transforms

from face_det.scrfd.detector import SCRFD
from face_rec.arcface.model import iresnet_inference
from face_rec.arcface.utils import read_features

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detector and recognizer
detector = SCRFD(model_file=r"face_det\scrfd\weights\scrfd_2.5g_bnkps.onnx")
recognizer = iresnet_inference(
    model_name="r100",
    path=r"face_rec\arcface\weights\arcface_r100.pth",
    device=device
)

@torch.no_grad()
def get_feature(face_image):
    """Extract facial features using GPU acceleration."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    
    # Get embeddings using GPU
    emb_img_face = recognizer(face_image)[0].cpu().numpy()
    
    # Convert to CuPy array and normalize
    emb_img_face = cp.array(emb_img_face)
    images_emb = emb_img_face / cp.linalg.norm(emb_img_face)
    
    return images_emb

def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """Add new persons to the database using GPU acceleration."""
    images_name = []
    images_emb = []

    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "JPG", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))
                bboxes, landmarks = detector.detect(image=input_image)

                for i in range(len(bboxes)):
                    number_files = len(os.listdir(person_face_path))
                    x1, y1, x2, y2, score = bboxes[i]
                    face_image = input_image[y1:y2, x1:x2]
                    
                    path_save_face = os.path.join(person_face_path, f"{number_files}.JPG")
                    cv2.imwrite(path_save_face, face_image)

                    # Get GPU-accelerated features
                    images_emb.append(cp.asnumpy(get_feature(face_image=face_image)))
                    images_name.append(name_person)

    if not images_emb:
        print("No new person found!")
        return None

    # Convert lists to arrays
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    # Handle existing features
    features = read_features(features_path)
    if features is not None:
        old_images_name, old_images_emb = features
        # Move old embeddings to CPU if they're on GPU
        if isinstance(old_images_emb, cp.ndarray):
            old_images_emb = cp.asnumpy(old_images_emb)
            
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        print("Update features!")

    # Save features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move new person data to backup
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)

    print("Successfully added new person!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=r"datasets\backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default=r"datasets\new_person",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default=r"datasets\data",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=r"datasets\face_features",
        help="Path to save face features.",
    )
    opt = parser.parse_args()
    
    add_persons(**vars(opt))
