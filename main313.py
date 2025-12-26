# main31_interactive.py
#
# Integrated interactive diagnostic GUI:
# - GWSF preprocessing
# - SFWS segmentation
# - APSIO feature selection (run on dataset during "Load & Prepare Dataset")
# - HCICNN training and classification
# - YOLO detection (optional) / contour fallback
# - GUI shows (1) enhanced image, (2) segmentation mask, (3) APSIO convergence plot, (4) HCICNN classification result, (5) YOLO overlay
# - Buttons for each stage + Run Full Flow + Save All Outputs
#
# Requirements: tensorflow==2.20.0, numpy, pandas, scikit-learn, opencv-python, pillow, matplotlib, seaborn
# Optional: ultralytics (YOLOv8)
#
# Usage:
# python main31_interactive.py --data_dir "<path_to_archive (2)>" --labels "<path_to_metadata.csv>" --max_images 2300
#

import os
import sys
import argparse
import logging
import json
import math
import random
import warnings
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # for saving plots in background; GUI uses FigureCanvasTkAgg with separate import
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Optional packages
try:
    import pywt
except ImportError:
    pywt = None
    print("Warning: pywt not installed — wavelet steps will use fallback smoothing.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    print("ultralytics YOLO not available — will use contour-based bounding boxes for visualization.")

# GUI-specific imports
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Logging and warnings
logging.basicConfig(level=logging.INFO, filename='main31_interactive.log', filemode='a',
                    format='%(asctime)s %(levelname)s: %(message)s')
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -------------------------
# APSIO (Adaptive Particle Swarm Intelligent Optimization) for feature selection
# -------------------------
class APSIO:
    def __init__(self, n_particles=20, max_iter=30, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def adaptive_inertia(self, t):
        return self.w_max - (self.w_max - self.w_min) * (t / max(1, (self.max_iter - 1)))

    def optimize(self, X: np.ndarray, y: np.ndarray, fitness_fn):
        n_features = X.shape[1]
        positions = np.random.rand(self.n_particles, n_features)
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, -np.inf)
        gbest_position = positions[0].copy()
        gbest_score = -np.inf
        convergence = []

        for it in range(self.max_iter):
            w = self.adaptive_inertia(it)
            for i in range(self.n_particles):
                mask = positions[i] > 0.5
                if np.sum(mask) == 0:
                    score = 0
                else:
                    try:
                        score = fitness_fn(X[:, mask], y)
                    except Exception as e:
                        logging.warning(f"APSIO fitness evaluation error: {e}")
                        score = 0
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()
            # update velocities & positions
            for i in range(self.n_particles):
                r1 = np.random.rand(n_features)
                r2 = np.random.rand(n_features)
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                transfer = 1 / (1 + np.exp(-velocities[i]))
                positions[i] = (transfer > np.random.rand(n_features)).astype(float)
            convergence.append(gbest_score)
            logging.info(f"APSIO iter {it+1}/{self.max_iter} best={gbest_score:.4f}")
        mask = gbest_position > 0.5
        if np.sum(mask) == 0:
            mask[np.argmax(np.abs(gbest_position))] = True
        return mask, gbest_score, convergence

# -------------------------
# GWSF (Gaussian Wavelet Spectral Filter)
# -------------------------
class GWSF:
    def __init__(self, sigma=0.6, wavelet='db4', enhancement_factor=1.6):
        self.sigma = sigma
        self.wavelet = wavelet
        self.enh = enhancement_factor

    def apply(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image_bgr is None:
            raise ValueError("None image to GWSF")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), int(max(1, self.sigma*3)))
        if pywt is not None:
            try:
                coeffs = pywt.wavedec2(blurred.astype(np.float32), self.wavelet, level=2)
                cA = coeffs[0]
                details = coeffs[1:]
                enhanced = [cA]
                for d in details:
                    if isinstance(d, tuple):
                        enhanced.append(tuple([self.enh * arr for arr in d]))
                    else:
                        enhanced.append(self.enh * d)
                recon = pywt.waverec2(enhanced, self.wavelet)
                recon = np.clip(recon, 0, 255).astype(np.uint8)
            except Exception as e:
                logging.warning(f"pywt recon failed: {e}")
                recon = blurred
        else:
            recon = blurred
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final = clahe.apply(recon)
        edges = cv2.Canny(final, 50, 150)
        if np.max(edges) > 0:
            edges = (edges.astype(np.float32)/edges.max() * 255).astype(np.uint8)
        return final, edges

# -------------------------
# SFWS (Slice Fragment Window Segmentation)
# -------------------------
class SFWS:
    def __init__(self, window_size=64, stride=32):
        self.window_size = window_size
        self.stride = stride

    def segment(self, gray_image: np.ndarray) -> np.ndarray:
        h, w = gray_image.shape
        mask = np.zeros((h,w), dtype=np.uint8)
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                y2 = min(y + self.window_size, h)
                x2 = min(x + self.window_size, w)
                win = gray_image[y:y2, x:x2]
                if win.size == 0:
                    continue
                var = np.var(win)
                m = np.mean(win)
                if var > 400 and m < 150:
                    mask[y:y2, x:x2] = 255
                else:
                    _, ots = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if np.mean(ots) < 250:
                        mask[y:y2, x:x2] = np.maximum(mask[y:y2, x:x2], (ots>0).astype(np.uint8)*255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

# -------------------------
# Feature extractor
# -------------------------
class FeatureExtractor:
    def __init__(self, target_length=100):
        self.target_length = target_length

    def extract(self, image_bgr: np.ndarray, mask: Optional[np.ndarray]=None, meta: Optional[dict]=None) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        features = []
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        angles = (np.arctan2(gy, gx) + np.pi) / (2*np.pi)
        ang_hist, _ = np.histogram(angles.flatten(), bins=12, range=(0,1))
        features.extend((ang_hist / (np.sum(ang_hist)+1e-6)).tolist())
        features.extend([mag.mean(), mag.std()])
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        for ch in range(3):
            c = hsv[:,:,ch]
            features.extend([float(np.mean(c)), float(np.std(c)), float(np.median(c))])
        features.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        features.append(float(np.std(cv2.boxFilter(gray.astype(float), -1, (3,3)))))
        if mask is not None and np.max(mask) > 0:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perim = cv2.arcLength(cnt, True)
                circ = 4*np.pi*area/(perim*perim + 1e-6) if perim>0 else 0
                features.extend([float(area), float(perim), float(circ)])
            else:
                features.extend([0.,0.,0.])
        else:
            features.extend([0.,0.,0.])
        if meta is not None:
            for k in ['age', 'diameter_1', 'diameter_2']:
                features.append(float(meta.get(k, 0.0) if meta.get(k, None) is not None else 0.0))
        else:
            features.extend([0.0,0.0,0.0])
        feats = np.array(features, dtype=np.float32)
        if feats.size < self.target_length:
            pad = np.zeros(self.target_length - feats.size, dtype=np.float32)
            feats = np.concatenate([feats, pad])
        else:
            feats = feats[:self.target_length]
        return feats

# -------------------------
# Small HCICNN builder
# -------------------------
def build_hcicnn(input_shape=(224,224,3), num_classes=6):
    inputs = Input(shape=input_shape)
    def hyper_conv_block(x, filters):
        c1 = Conv2D(filters//2, (3,3), padding='same', activation='relu')(x)
        c2 = Conv2D(filters//2, (5,5), padding='same', activation='relu')(x)
        merged = layers.Concatenate()([c1, c2])
        merged = BatchNormalization()(merged)
        merged = layers.Activation('relu')(merged)
        merged = MaxPooling2D((2,2))(merged)
        return merged
    x = hyper_conv_block(inputs, 64)
    x = hyper_conv_block(x, 128)
    x = hyper_conv_block(x, 256)
    x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Dataset loader (auto-detect)
# -------------------------
class DatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.expanduser(data_dir)

    def find_all_image_paths(self, valid_exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
        image_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(root, f))
        return image_paths

    def load(self, labels_csv: str, sample_fraction=1.0, min_per_class=30):
        labels_csv = os.path.expanduser(labels_csv)
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(labels_csv)
        df = pd.read_csv(labels_csv)
        if 'img_id' not in df.columns or 'diagnostic' not in df.columns:
            raise ValueError("CSV must contain 'img_id' and 'diagnostic'")
        print("\nOriginal class distribution:")
        print(df['diagnostic'].value_counts())
        if df['diagnostic'].nunique() <= 1:
            raise ValueError("Only one class in CSV")
        balanced_frames = []
        class_dist = df['diagnostic'].value_counts()
        max_count = int(class_dist.max() * sample_fraction)
        for cls, cnt in class_dist.items():
            group = df[df['diagnostic'] == cls]
            target = max(min_per_class, min(len(group), max_count))
            sampled = group.sample(n=target, replace=(len(group)<target), random_state=SEED)
            balanced_frames.append(sampled)
        df_bal = pd.concat(balanced_frames, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        print(f"Balanced df created: {len(df_bal)} samples across {df_bal['diagnostic'].nunique()} classes")
        all_images = self.find_all_image_paths()
        if len(all_images) == 0:
            raise ValueError("No images found under data_dir")
        lookup = {}
        for p in all_images:
            base = os.path.splitext(os.path.basename(p))[0]
            lookup.setdefault(base, []).append(p)
        images, labels, paths = [], [], []
        for _, row in df_bal.iterrows():
            img_id = str(row['img_id']).split('.')[0]
            cls = row['diagnostic']
            cand = lookup.get(img_id, [])
            if not cand:
                # fuzzy match: endswith
                p_match = None
                for b, lst in lookup.items():
                    if b.endswith(img_id):
                        p_match = lst[0]
                        break
                if p_match is None:
                    continue
                p = p_match
            else:
                p = cand[0]
            im = cv2.imread(p)
            if im is None:
                continue
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
            im_resized = cv2.resize(im, (512,512), interpolation=cv2.INTER_AREA)
            images.append(im_resized); labels.append(cls); paths.append(p)
        if len(images) == 0:
            raise ValueError("No images matched labels")
        print(f"Loaded {len(images)} images.")
        print(pd.Series(labels).value_counts())
        return images, labels, paths, df_bal

# -------------------------
# Pipeline class (holds trained models, selected_mask, convergence)
# -------------------------
class SkinDiseasePipeline:
    def __init__(self, data_dir, output_dir='./interactive_output', img_size=(224,224)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.loader = DatasetLoader(self.data_dir)
        self.gwsf = GWSF()
        self.sfws = SFWS()
        self.fe = FeatureExtractor(target_length=100)
        self.apsio = APSIO(n_particles=20, max_iter=30)
        self.hcicnn = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.img_size = img_size
        self.yolo = None
        if ULTRALYTICS_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
            except Exception as e:
                logging.warning(f"YOLO init failed: {e}")
                self.yolo = None
        self.class_names = None
        self.hcicnn_model_path = os.path.join(self.output_dir, 'hcicnn_interactive.h5')
        self.selected_mask = None
        self.apsio_convergence = None
        self.apsio_best_score = None

    def load_and_train_full(self, labels_csv, max_images=None, epochs=20, batch_size=16):
        """Loads dataset, extracts features, runs APSIO, trains HCICNN. Stores selected_mask & convergence."""
        images, labels, img_paths, df_bal = self.loader.load(labels_csv)
        if max_images is not None:
            images = images[:max_images]; labels = labels[:max_images]; img_paths = img_paths[:max_images]
        y_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = list(self.label_encoder.classes_)
        # process images: GWSF + SFWS + features
        processed = []; masks = []; feature_maps = []; features = []
        for i, (img, p) in enumerate(zip(images, img_paths)):
            if i % 50 == 0:
                print(f"Processing {i+1}/{len(images)}")
            enhanced, fmap = self.gwsf.apply(img)
            mask = self.sfws.segment(enhanced)
            feat = self.fe.extract(img, mask, None)
            processed.append(enhanced); masks.append(mask); feature_maps.append(fmap); features.append(feat)
        features = np.vstack(features).astype(np.float32)
        if features.size == 0 or features.ndim != 2:
            raise ValueError("No features extracted")
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        self.images = images; self.processed_images = processed; self.masks = masks; self.feature_maps = feature_maps
        self.features = features_scaled; self.y = y_encoded; self.df_bal = df_bal
        # APSIO
        def fitness(X_sel,y):
            if X_sel.shape[1] == 0 or len(np.unique(y)) <= 1:
                return 0
            try:
                clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
                folds = min(3, max(2, len(np.unique(y))))
                scores = cross_val_score(clf, X_sel, y, cv=folds, scoring='accuracy')
                return float(np.mean(scores))
            except Exception as e:
                logging.warning(f"Fitness error: {e}"); return 0
        mask, best_score, conv = self.apsio.optimize(self.features, self.y, fitness)
        self.selected_mask = mask; self.apsio_best_score = best_score; self.apsio_convergence = conv
        print(f"APSIO selected {np.sum(mask)} features best={best_score:.4f}")
        # Train traditional models optionally (skipped in GUI for speed)
        # Prepare images for HCICNN
        X = []
        for im in self.processed_images:
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if len(im.shape)==3 else cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            im_res = cv2.resize(im_rgb, self.img_size, interpolation=cv2.INTER_AREA)
            X.append(im_res.astype(np.float32)/255.0)
        X = np.array(X, dtype=np.float32)
        y = np.array(self.y, dtype=np.int32)
        # shuffle
        idx = np.arange(len(X)); np.random.shuffle(idx); X=X[idx]; y=y[idx]
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(self.class_names))
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=SEED, stratify=y)
        # model
        self.hcicnn = build_hcicnn(input_shape=(self.img_size[0], self.img_size[1], 3), num_classes=len(self.class_names))
        datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                     zoom_range=0.1, horizontal_flip=True, fill_mode='reflect')
        datagen.fit(X_train)
        y_integers = np.argmax(y_train, axis=1)
        class_weights_values = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
        class_weights = {i: w for i, w in enumerate(class_weights_values)}
        cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)]
        history = self.hcicnn.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  validation_data=(X_test, y_test),
                                  epochs=epochs, callbacks=cb, class_weight=class_weights, verbose=1)
        loss, acc = self.hcicnn.evaluate(X_test, y_test, verbose=0)
        self.hcicnn.save(self.hcicnn_model_path)
        # save APSIO convergence plot
        try:
            plt.figure(figsize=(6,4)); plt.plot(self.apsio_convergence, marker='o'); plt.title('APSIO Convergence'); plt.xlabel('Iteration'); plt.ylabel('Fitness'); plt.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'apsio_convergence.png'), dpi=200); plt.close()
        except Exception as e:
            logging.warning(f"Saving APSIO convergence failed: {e}")
        return history, (loss, acc)

    def predict_single_image(self, img_bgr):
        enhanced, fmap = self.gwsf.apply(img_bgr)
        mask = self.sfws.segment(enhanced)
        feat = self.fe.extract(img_bgr, mask, None)
        feat_scaled = self.scaler.transform(feat.reshape(1,-1)) if hasattr(self, 'scaler') else feat.reshape(1,-1)
        selected_features = feat_scaled[0, self.selected_mask] if (self.selected_mask is not None and feat_scaled.shape[1] >= len(self.selected_mask)) else None
        # prepare for HCICNN
        im_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if len(enhanced.shape)==3 else cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        im_res = cv2.resize(im_rgb, self.img_size)
        x = (im_res.astype(np.float32)/255.0)[None,...]
        if self.hcicnn is None:
            if os.path.exists(self.hcicnn_model_path):
                try:
                    self.hcicnn = tf.keras.models.load_model(self.hcicnn_model_path)
                except Exception as e:
                    logging.error(f"Load model failed: {e}")
        if self.hcicnn is None:
            raise RuntimeError("HCICNN model not available. Run Load & Prepare Dataset first.")
        pred = self.hcicnn.predict(x, verbose=0)[0]
        cls_idx = int(np.argmax(pred)); label = self.class_names[cls_idx] if self.class_names is not None else str(cls_idx); conf = float(pred[cls_idx])
        # YOLO detection / contour fallback on enhanced image (we will return bbox relative to enhanced 512x512)
        bbox = None
        if self.yolo is not None:
            try:
                tmp = os.path.join(self.output_dir, 'tmp_img.jpg')
                cv2.imwrite(tmp, img_bgr)
                results = self.yolo(tmp)
                if results and len(results) > 0:
                    try:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        scores = results[0].boxes.conf.cpu().numpy()
                        if len(boxes) > 0:
                            best = np.argmax(scores)
                            bb = boxes[best].astype(int).tolist()
                            bbox = bb
                    except Exception:
                        bbox = None
            except Exception as e:
                logging.warning(f"YOLO detect error: {e}")
        if bbox is None:
            # fallback to largest contour in mask (mask size is enhanced (512x512) because enhanced was created from original which we resized earlier to 512)
            try:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    x,y,w,h = cv2.boundingRect(cnt)
                    bbox = [int(x), int(y), int(x+w), int(y+h)]
            except Exception:
                bbox = None
        return {
            'enhanced': enhanced,
            'feature_map': fmap,
            'mask': mask,
            'features': feat,
            'features_selected': selected_features,
            'label': label,
            'confidence': conf,
            'bbox': bbox
        }

# -------------------------
# GUI: Scrollable 5-panel layout with buttons & save outputs
# -------------------------
class InteractiveGUI:
    def __init__(self, pipeline: SkinDiseasePipeline, labels_csv: str, max_images: Optional[int]=None):
        self.pipeline = pipeline
        self.labels_csv = labels_csv
        self.max_images = max_images
        self.current_bgr = None
        self.last_result = None
        self.root = tk.Tk()
        self.root.title("Interactive Diagnostic Flow (GWSF → SFWS → APSIO → HCICNN → YOLO)")
        self.root.geometry("1400x900")
        self._build_ui()

    def _build_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        btn_cfg = {'width':20, 'height':1, 'padx':4, 'pady':4}

        self.load_btn = tk.Button(control_frame, text="Load & Prepare Dataset (Run APSIO & Train HCICNN)",
                                   command=self._run_full_pipeline, bg='#4CAF50', fg='white', **btn_cfg)
        self.load_btn.pack(side=tk.LEFT, padx=6)

        self.select_btn = tk.Button(control_frame, text="Select Image", command=self._select_image, bg='#2196F3', fg='white', **btn_cfg)
        self.select_btn.pack(side=tk.LEFT, padx=6)

        self.gwsf_btn = tk.Button(control_frame, text="Apply GWSF", command=self._apply_gwsf, state=tk.DISABLED, bg='#FF9800', fg='white', **btn_cfg)
        self.gwsf_btn.pack(side=tk.LEFT, padx=6)

        self.sfws_btn = tk.Button(control_frame, text="Apply SFWS", command=self._apply_sfws, state=tk.DISABLED, bg='#9C27B0', fg='white', **btn_cfg)
        self.sfws_btn.pack(side=tk.LEFT, padx=6)

        self.apsio_btn = tk.Button(control_frame, text="Show APSIO (selected features)", command=self._show_apsio, state=tk.DISABLED, bg='#E91E63', fg='white', **btn_cfg)
        self.apsio_btn.pack(side=tk.LEFT, padx=6)

        self.classify_btn = tk.Button(control_frame, text="Classify (HCICNN)", command=self._classify_image, state=tk.DISABLED, bg='#00BCD4', fg='white', **btn_cfg)
        self.classify_btn.pack(side=tk.LEFT, padx=6)

        self.yolo_btn = tk.Button(control_frame, text="YOLO Detect (or Contour)", command=self._yolo_overlay, state=tk.DISABLED, bg='#795548', fg='white', **btn_cfg)
        self.yolo_btn.pack(side=tk.LEFT, padx=6)

        self.fullflow_btn = tk.Button(control_frame, text="Run Full Flow (All Steps)", command=self._run_full_flow_for_selected, bg='#3F51B5', fg='white', **btn_cfg)
        self.fullflow_btn.pack(side=tk.LEFT, padx=6)

        self.save_btn = tk.Button(control_frame, text="Save All Outputs (for current image)", command=self._save_all_outputs, state=tk.DISABLED, bg='#607D8B', fg='white', **btn_cfg)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # Status label
        self.status_var = tk.StringVar(value="Status: Ready")
        status_lbl = tk.Label(self.root, textvariable=self.status_var, anchor='w', justify='left')
        status_lbl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        # Scrollable canvas for 5 panels
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Now add 5 panels vertically
        panel_titles = [
            "1) Original Image",
            "2) GWSF Enhanced Image",
            "3) SFWS Segmentation Mask (isolated lesion)",
            "4) APSIO Convergence & Feature Summary",
            "5) HCICNN Classification + YOLO Overlay"
        ]
        self.panel_images = []
        self.panel_canvases = []
        self.matplot_canvas = None  # for APSIO convergence

        for title in panel_titles:
            frame = tk.LabelFrame(self.scrollable_frame, text=title, padx=6, pady=6, font=('Arial', 11, 'bold'))
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            img_lbl = tk.Label(frame)
            img_lbl.pack(fill=tk.BOTH, expand=True)
            self.panel_images.append(img_lbl)

        # Add a small text area for APSIO summary (below the third panel)
        self.apsio_summary_var = tk.StringVar(value="APSIO summary: Not available")
        self.apsio_summary_lbl = tk.Label(self.scrollable_frame, textvariable=self.apsio_summary_var, anchor='w', justify='left', font=('Arial', 10))
        self.apsio_summary_lbl.pack(fill=tk.X, padx=12, pady=(0,8))

    def _update_status(self, msg):
        self.status_var.set(f"Status: {msg}")
        self.root.update()

    def _run_full_pipeline(self):
        # runs data load -> APSIO -> HCICNN training. This must be done once before classification gets accurate.
        try:
            self._update_status("Loading dataset and training... (this may take several minutes)")
            self.pipeline.load_and_train_full(self.labels_csv, max_images=self.max_images, epochs=20, batch_size=16)
            self._update_status("Dataset prepared, APSIO run, HCICNN trained and saved.")
            # enable per-image buttons
            self.gwsf_btn.config(state=tk.NORMAL)
            self.sfws_btn.config(state=tk.NORMAL)
            self.apsio_btn.config(state=tk.NORMAL)
            self.classify_btn.config(state=tk.NORMAL)
            self.yolo_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Pipeline", "Dataset prepared and model trained. You can now select images and run steps.")
        except Exception as e:
            logging.error(f"Full pipeline error: {e}")
            messagebox.showerror("Error", f"Full pipeline failed: {e}")
            self._update_status("Full pipeline failed. See logs.")

    def _select_image(self):
        p = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        # Resize original for display 512x512 for consistency
        display = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        self.current_bgr = display.copy()
        self._show_in_panel(0, display)
        self._update_status(f"Selected image: {os.path.basename(p)}")
        # enable step buttons only if pipeline has been trained
        # some operations (GWSF,SFWS) can be applied even if not trained, but APSIO/classify require training
        self.gwsf_btn.config(state=tk.NORMAL)
        self.sfws_btn.config(state=tk.NORMAL)
        if self.pipeline.hcicnn is not None and self.pipeline.selected_mask is not None:
            self.apsio_btn.config(state=tk.NORMAL)
            self.classify_btn.config(state=tk.NORMAL)
            self.yolo_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

    def _show_in_panel(self, idx, img_rgb_or_gray):
        # idx: 0..4 corresponds to panels; img is RGB or gray (BGR expected for incoming)
        if img_rgb_or_gray is None:
            return
        # Ensure RGB array for PIL
        arr = img_rgb_or_gray.copy()
        if len(arr.shape) == 2:
            mode = 'L'
            im = Image.fromarray(arr)
        else:
            # if BGR -> convert to RGB
            if arr.shape[2] == 3:
                im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            else:
                im = Image.fromarray(arr)
        imtk = ImageTk.PhotoImage(im)
        lbl = self.panel_images[idx]
        lbl.config(image=imtk)
        lbl.image = imtk

    def _apply_gwsf(self):
        if self.current_bgr is None:
            messagebox.showerror("Error", "Select an image first")
            return
        enhanced, fmap = self.pipeline.gwsf.apply(self.current_bgr)
        # enhanced is grayscale (512x512) — show as RGB for consistency
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        self._show_in_panel(1, enhanced_rgb)
        self.last_result = {'enhanced': enhanced, 'feature_map': fmap}
        self._update_status("GWSF applied. Enhanced image shown.")
        # enable SFWS
        self.sfws_btn.config(state=tk.NORMAL)

    def _apply_sfws(self):
        if not self.last_result or 'enhanced' not in self.last_result:
            messagebox.showerror("Error", "Run GWSF first")
            return
        enhanced = self.last_result['enhanced']
        mask = self.pipeline.sfws.segment(enhanced)
        # Mask overlay: isolate lesion on original image
        # If current_bgr exists (512 size), overlay mask on it
        overlay = self.current_bgr.copy()
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
        self._show_in_panel(2, overlay)
        self.last_result['mask'] = mask
        self._update_status("SFWS applied. Segmentation mask shown.")
        # enable APSIO summary & classification if pipeline has APSIO done
        if self.pipeline.selected_mask is not None:
            self.apsio_btn.config(state=tk.NORMAL)
            self.classify_btn.config(state=tk.NORMAL)
            self.yolo_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

    def _show_apsio(self):
        # Show APSIO convergence and summary in panel 3 (we'll place the convergence plot in the same area).
        if self.pipeline.apsio_convergence is None:
            messagebox.showerror("Error", "APSIO not run yet. Use Load & Prepare Dataset first.")
            return
        conv = self.pipeline.apsio_convergence
        fig = plt.Figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        ax.plot(conv, marker='o'); ax.set_title("APSIO Convergence"); ax.set_xlabel("Iteration"); ax.set_ylabel("Fitness")
        ax.grid(True)
        # display matplotlib figure in an image and show in panel 3
        # save to temporary image
        tmp_plot = os.path.join(self.pipeline.output_dir, 'tmp_apsio_plot.png')
        fig.savefig(tmp_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        img = cv2.imread(tmp_plot)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._show_in_panel(3, img)
        # update summary string showing selected features count and best score
        selected_count = int(np.sum(self.pipeline.selected_mask)) if self.pipeline.selected_mask is not None else 0
        best_score = float(self.pipeline.apsio_best_score) if self.pipeline.apsio_best_score is not None else 0.0
        self.apsio_summary_var.set(f"APSIO summary: selected features = {selected_count}, best fitness = {best_score:.4f}")
        self._update_status("APSIO convergence plot shown and summary updated.")
        # enable classify & save
        self.classify_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

    def _classify_image(self):
        if self.current_bgr is None:
            messagebox.showerror("Error", "Select an image first")
            return
        try:
            res = self.pipeline.predict_single_image(self.current_bgr)
            self.last_result.update(res)
            # show classification results in panel 4 (we will display small image with text)
            # create a visualization: enhanced with label text
            enhanced = res['enhanced']
            if len(enhanced.shape) == 2:
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_rgb = enhanced
            disp = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2RGB).copy()
            label = res['label']; conf = res['confidence']
            txt = f"{label} ({conf*100:.1f}%)"
            cv2.putText(disp, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            self._show_in_panel(4, disp)
            self._update_status(f"HCICNN predicted: {txt}")
            # also show APSIO selected features length in panel 3 summary if available
            if res.get('features_selected') is not None:
                sel_len = res['features_selected'].shape[0]
                self.apsio_summary_var.set(f"APSIO summary: selected features = {sel_len}, best fitness = {self.pipeline.apsio_best_score:.4f}")
            else:
                self.apsio_summary_var.set("APSIO: selected features not available for this image.")
            self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(f"Classify image error: {e}")
            messagebox.showerror("Error", f"Classification failed: {e}")

    def _yolo_overlay(self):
        if self.last_result is None or 'enhanced' not in self.last_result:
            messagebox.showerror("Error", "Run GWSF and SFWS first (or Run Full Flow) before YOLO overlay.")
            return
        enhanced = self.last_result['enhanced']
        overlay_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(enhanced.shape)==2 else enhanced.copy()
        bbox = self.last_result.get('bbox', None)
        if bbox is not None:
            try:
                x1,y1,x2,y2 = bbox
                # ensure within bounds
                h,w = overlay_rgb.shape[:2]
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                cv2.rectangle(overlay_rgb, (x1,y1), (x2,y2), (255,0,0), 3)
            except Exception:
                pass
        else:
            # try draw contour mask
            mask = self.last_result.get('mask', None)
            if mask is not None:
                color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay_rgb = cv2.addWeighted(cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB), 0.7, color_mask, 0.3, 0)
        self._show_in_panel(4, overlay_rgb)
        self._update_status("YOLO overlay (or contour) shown on enhanced image.")

    def _run_full_flow_for_selected(self):
        # convenience: if user selected an image before training, run the training silently then apply all steps for the selected image
        if self.current_bgr is None:
            messagebox.showerror("Error", "Select an image first")
            return
        if self.pipeline.hcicnn is None or self.pipeline.selected_mask is None:
            # run full pipeline
            self._run_full_pipeline()
        # now run GWSF -> SFWS -> Show APSIO -> Classify -> YOLO overlay in sequence
        self._apply_gwsf()
        self._apply_sfws()
        self._show_apsio()
        self._classify_image()
        self._yolo_overlay()
        self._update_status("Full flow executed for selected image.")

    def _save_all_outputs(self):
        if self.current_bgr is None or self.last_result is None:
            messagebox.showerror("Error", "No current image results to save")
            return
        base_name = f"result_{int(pd.Timestamp.now().timestamp())}"
        out_dir = os.path.join(self.pipeline.output_dir, base_name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            # save original
            cv2.imwrite(os.path.join(out_dir, 'original.jpg'), self.current_bgr)
            # enhanced
            if 'enhanced' in self.last_result and self.last_result['enhanced'] is not None:
                enh = self.last_result['enhanced']
                enh_save = enh if len(enh.shape)==3 else cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(out_dir, 'enhanced.jpg'), enh_save)
            # mask
            if 'mask' in self.last_result and self.last_result['mask'] is not None:
                cv2.imwrite(os.path.join(out_dir, 'mask.png'), self.last_result['mask'])
                # overlay
                overlay = cv2.addWeighted(self.current_bgr, 0.7, cv2.cvtColor(self.last_result['mask'], cv2.COLOR_GRAY2BGR), 0.3, 0)
                cv2.imwrite(os.path.join(out_dir, 'overlay.jpg'), overlay)
            # features
            if 'features' in self.last_result and self.last_result['features'] is not None:
                np.save(os.path.join(out_dir, 'features.npy'), self.last_result['features'])
            if 'features_selected' in self.last_result and self.last_result['features_selected'] is not None:
                np.save(os.path.join(out_dir, 'features_selected.npy'), self.last_result['features_selected'])
            # classification
            cls_txt = f"label: {self.last_result.get('label','NA')}\nconfidence: {self.last_result.get('confidence',0.0):.4f}\n"
            with open(os.path.join(out_dir, 'classification.txt'), 'w') as f:
                f.write(cls_txt)
            # APSIO convergence plot copy if exists in pipeline.output_dir
            src_plot = os.path.join(self.pipeline.output_dir, 'apsio_convergence.png')
            if os.path.exists(src_plot):
                import shutil
                shutil.copy(src_plot, os.path.join(out_dir, 'apsio_convergence.png'))
            self._update_status(f"Saved outputs to {out_dir}")
            messagebox.showinfo("Saved", f"All outputs saved to {out_dir}")
        except Exception as e:
            logging.error(f"Save outputs error: {e}")
            messagebox.showerror("Error", f"Failed to save outputs: {e}")

    def run(self):
        self.root.mainloop()

# -------------------------
# Command-line interface and start
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive diagnostic GUI for skin disease pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root (scanned recursively)')
    parser.add_argument('--labels', type=str, required=True, help='Metadata CSV (img_id, diagnostic)')
    parser.add_argument('--output_dir', type=str, default='./interactive_output', help='Output directory')
    parser.add_argument('--max_images', type=int, default=None, help='Max images for dataset preparation')
    args = parser.parse_args()

    print("Interactive Diagnostic GUI starting...")
    pipeline = SkinDiseasePipeline(data_dir=args.data_dir, output_dir=args.output_dir)
    gui = InteractiveGUI(pipeline, labels_csv=args.labels, max_images=args.max_images)
    gui.run()
