import threading
import queue
import time
import cv2
import cupy as cp
import numpy as np
import torch
from torchvision import transforms
from typing import List, Tuple
from dataclasses import dataclass

from face_rec.arcface.model import iresnet_inference
from face_rec.arcface.utils import compare_encodings, read_features
from face_alignment.alignment import norm_crop
from face_det.scrfd.detector import SCRFD

@dataclass
class DetectedFace:
    bbox: np.ndarray
    landmarks: np.ndarray
    aligned_face: np.ndarray
    confidence: float

class FaceRecognitionSystem:
    def __init__(
        self,
        arcface_model_path: str,
        feature_path: str,
        scrfd_model_path: str,
        batch_size: int = 8,
        queue_size: int = 30,
        detection_threshold: float = 0.5,
        recognition_threshold: float = 0.5
    ):
        # Initialize CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize queues
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)

        # Configuration
        self.batch_size = batch_size
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold

        # Initialize SCRFD detector
        self.detector = SCRFD(model_file=scrfd_model_path)

        # Initialize ArcFace recognizer
        self.recognizer = iresnet_inference(
            model_name="r100",
            path=arcface_model_path,
            device=self.device
        )

        # Load features
        features = read_features(feature_path)
        if features is None:
            raise ValueError("Could not load features from the specified path")
        self.images_names, self.images_embs = features

        # Threading
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)

        self.is_running = threading.Event()

    @torch.no_grad()
    def get_features(self, face_images: List[np.ndarray]) -> cp.ndarray:
        """Extract features from a batch of face images."""
        face_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        processed_faces = torch.stack([
            face_preprocess(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            for face in face_images
        ]).to(self.device)

        embeddings = self.recognizer(processed_faces).cpu().numpy()

        emb_gpu = cp.array(embeddings)
        emb_gpu = emb_gpu / cp.linalg.norm(emb_gpu, axis=1, keepdims=True)

        return emb_gpu

    def _detection_worker(self):
        """Worker thread for face detection."""
        while self.is_running.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                outputs, img_info, bboxes, landmarks = self.detector.detect_tracking(image=frame)

                detected_faces = []
                if outputs is not None:
                    for i in range(len(bboxes)):
                        x1, y1, x2, y2, score = bboxes[i]
                        if score < self.detection_threshold:
                            continue

                        face_image = frame[int(y1):int(y2), int(x1):int(x2)]
                        aligned_face = norm_crop(frame, landmarks[i])

                        detected_faces.append(DetectedFace(
                            bbox=np.array([x1, y1, x2, y2]),
                            landmarks=landmarks[i],
                            aligned_face=aligned_face,
                            confidence=score
                        ))

                self.result_queue.put((frame, detected_faces))

            except Exception as e:
                print(f"Detection error: {e}")
            finally:
                self.frame_queue.task_done()

    def _recognition_worker(self):
        """Worker thread for face recognition."""
        while self.is_running.is_set():
            try:
                frame, detected_faces = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if detected_faces:
                    batch_faces = [face.aligned_face for face in detected_faces]
                    batch_features = self.get_features(batch_faces)

                    for face, feature in zip(detected_faces, batch_features):
                        score, idx = compare_encodings(feature[None, :], self.images_embs)
                        name = self.images_names[idx] if score >= self.recognition_threshold else "UNKNOWN"

                        x1, y1, x2, y2 = map(int, face.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name}: {score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

            except Exception as e:
                print(f"Recognition error: {e}")
            finally:
                self.result_queue.task_done()

    def start(self):
        self.is_running.set()
        self.detection_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.is_running.clear()
        self.detection_thread.join()
        self.recognition_thread.join()

    def process_frame(self, frame: np.ndarray):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

if __name__ == "__main__":
    face_recognition = FaceRecognitionSystem(
        arcface_model_path=r"D:\face_recognition\new\face_rec\arcface\weights\arcface_r100.pth",
        feature_path=r"D:\face_recognition\new\datasets\face_features",
        scrfd_model_path=r"D:\face_recognition\new\face_det\scrfd\weights\scrfd_2.5g_bnkps.onnx"
    )

    face_recognition.start()

    cap = cv2.VideoCapture("http://192.168.137.65:4747/video")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_recognition.process_frame(frame)

    face_recognition.stop()
    cap.release()
    cv2.destroyAllWindows()