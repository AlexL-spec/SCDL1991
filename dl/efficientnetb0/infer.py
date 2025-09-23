# src/dl/efficientnetb0/infer.py
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2
from skimage.color import rgb2hed
import PIL.Image
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'


def ensure_pil_rgb(image_input):
    if isinstance(image_input, (str, Path)):
        img_pil = PIL.Image.open(image_input)
    elif isinstance(image_input, PIL.Image.Image):
        img_pil = image_input
    else:
        raise ValueError("Input must be a file path or PIL.Image object.")
    if img_pil.mode != 'RGB':
        return img_pil.convert('RGB')
    return img_pil


def preprocess_for_inference(
        image_input,
        target_colorspace: str,
        target_img_size_hw: tuple = (224, 224),
        cnn_model_name: str = "efficientnetb0"
):
    pil_img_rgb = ensure_pil_rgb(image_input)

    if target_colorspace == "grayscale":
        processed_pil_img = pil_img_rgb.convert("L").convert("RGB")
    elif target_colorspace == "ycbcr":
        img_cv_bgr = cv2.cvtColor(np.array(pil_img_rgb), cv2.COLOR_RGB2BGR)
        img_ycbcr_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2YCrCb)
        img_rgb_restored = cv2.cvtColor(img_ycbcr_cv, cv2.COLOR_YCrCb2RGB)
        processed_pil_img = PIL.Image.fromarray(img_rgb_restored)
    elif target_colorspace == "cielab":
        img_cv_bgr = cv2.cvtColor(np.array(pil_img_rgb), cv2.COLOR_RGB2BGR)
        img_lab_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2Lab)
        img_rgb_restored = cv2.cvtColor(img_lab_cv, cv2.COLOR_Lab2RGB)
        processed_pil_img = PIL.Image.fromarray(img_rgb_restored)
    elif target_colorspace == "hed":
        img_rgb_sk = np.array(pil_img_rgb)
        try:
            ihc_hed = rgb2hed(img_rgb_sk)
            h = PIL.Image.fromarray((ihc_hed[:, :, 0] * 255).astype(np.uint8)).convert("L")
            e = PIL.Image.fromarray((ihc_hed[:, :, 1] * 255).astype(np.uint8)).convert("L")
            d = PIL.Image.fromarray((ihc_hed[:, :, 2] * 255).astype(np.uint8)).convert("L")
            processed_pil_img = PIL.Image.merge("RGB", (h, e, d))
        except Exception as e_hed:
            print(f"Warning: HED conversion failed for an image: {e_hed}. Using original RGB.")
            processed_pil_img = pil_img_rgb
    elif target_colorspace == "original" or target_colorspace == "rgb":
        processed_pil_img = pil_img_rgb
    else:
        print(f"Warning: Unsupported target_colorspace '{target_colorspace}'. Using original RGB for inference.")
        processed_pil_img = pil_img_rgb

    img_tf = tf.convert_to_tensor(np.array(processed_pil_img), dtype=tf.float32)
    img_tf = tf.image.resize(img_tf, target_img_size_hw)

    if cnn_model_name == "efficientnetb0":
        img_tf_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_tf)
    elif cnn_model_name == "mobilenetv2":
        img_tf_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_tf)
    elif cnn_model_name == "resnet50":
        img_tf_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_tf)
    else:
        print(
            f"Warning: No specific preprocessing for cnn_model_name '{cnn_model_name}'. Applying generic rescale 1./255.")
        img_tf_preprocessed = img_tf / 255.0

    return img_tf_preprocessed


def main():
    parser = argparse.ArgumentParser(
        description="Batch/single-image inference for trained EfficientNetB0 models.")
    parser.add_argument("--model_path", required=True, type=Path, help="Path to the trained .keras model file.")
    parser.add_argument("--input_path", required=True, type=Path,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("--output_csv", default="predictions_efficientnetb0.csv", type=Path,
                        help="Path to save the prediction results CSV.")
    parser.add_argument("--target_colorspace", type=str, required=True,
                        help="The colorspace the model was trained on (e.g., original, grayscale, etc.).")
    parser.add_argument("--img_height", type=int, default=224, help="Target image height for model input.")
    parser.add_argument("--img_width", type=int, default=224, help="Target image width for model input.")
    parser.add_argument("--class_map_json", type=Path, default=None,
                        help="Optional path to class_map.json (idx_to_class_name).")

    args = parser.parse_args()

    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
        print(f"EfficientNetB0 model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading EfficientNetB0 model: {e}")
        sys.exit(1)

    idx_to_class_name = None
    if args.class_map_json and args.class_map_json.exists():
        try:
            with open(args.class_map_json, 'r', encoding='utf-8') as f:
                idx_to_class_name_str_keys = json.load(f)
                idx_to_class_name = {int(k): v for k, v in idx_to_class_name_str_keys.items()}
            print(f"Class map loaded from {args.class_map_json}")
        except Exception as e:
            print(f"Warning: Could not load or parse class_map_json: {e}")

    image_files = []
    if args.input_path.is_file():
        image_files.append(args.input_path)
    elif args.input_path.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(args.input_path.glob(f"**/{ext}")))

    if not image_files:
        print(f"No images found in {args.input_path}")
        return

    print(f"Found {len(image_files)} images to predict using EfficientNetB0.")
    predictions_data = []
    target_size = (args.img_height, args.img_width)

    for img_file in tqdm(image_files, desc="Predicting images (EfficientNetB0)"):
        try:
            preprocessed_img_tensor = preprocess_for_inference(
                img_file,
                args.target_colorspace,
                target_size,
                cnn_model_name="efficientnetb0"  # Explicitly pass model name
            )
            batch_img_tensor = tf.expand_dims(preprocessed_img_tensor, axis=0)

            pred_probs = model.predict(batch_img_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(pred_probs))
            confidence = float(pred_probs[pred_idx])

            pred_class_name = str(pred_idx)
            if idx_to_class_name and pred_idx in idx_to_class_name:
                pred_class_name = idx_to_class_name[pred_idx]

            predictions_data.append({
                'image_path': str(img_file.name),
                'full_path': str(img_file.resolve()),
                'predicted_class_index': pred_idx,
                'predicted_class_name': pred_class_name,
                'confidence': confidence,
                'all_probabilities': pred_probs.tolist()
            })
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            predictions_data.append({
                'image_path': str(img_file.name),
                'full_path': str(img_file.resolve()),
                'predicted_class_index': -1,
                'predicted_class_name': 'Error',
                'confidence': 0.0,
                'all_probabilities': []
            })

    df_preds = pd.DataFrame(predictions_data)
    try:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"EfficientNetB0 predictions saved to {args.output_csv}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")


if __name__ == '__main__':
    main()