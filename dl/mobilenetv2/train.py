# src/dl/mobilenetv2/train.py
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

import matplotlib.pyplot as plt
import json
import time
import os
import sys

from skimage.color import rgb2gray, rgb2hed, rgb2ycbcr, rgb2lab
from skimage.exposure import rescale_intensity

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'


def load_run_config(config_path: Path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_custom_mobilenetv2_model(num_classes, img_size_hw=(224, 224), base_trainable_setting=15, dropout_rate=0.3):
    input_shape_with_channels = (*img_size_hw, 3)
    base_model_for_build = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape_with_channels,
        pooling="avg"
    )

    if isinstance(base_trainable_setting, bool):
        base_model_for_build.trainable = base_trainable_setting
        print(f"  MobileNetV2 base_model_for_build.trainable set to: {base_trainable_setting}")
    elif isinstance(base_trainable_setting, int) and base_trainable_setting >= 0:
        if base_trainable_setting == 0:
            print(f"  Freezing all layers of MobileNetV2 base_model_for_build.")
            base_model_for_build.trainable = False
        elif len(base_model_for_build.layers) >= base_trainable_setting:
            print(f"  Unfreezing last {base_trainable_setting} layers of MobileNetV2 base_model_for_build.")
            for layer in base_model_for_build.layers: layer.trainable = False
            for layer in base_model_for_build.layers[-base_trainable_setting:]: layer.trainable = True
        else:
            print(
                f"  Warning: base_trainable_setting ({base_trainable_setting}) >= num layers in MobileNetV2 base_model_for_build ({len(base_model_for_build.layers)}). Unfreezing all base_model_for_build layers.")
            base_model_for_build.trainable = True
    else:
        print(
            f"  Invalid base_trainable_setting: {base_trainable_setting}. Freezing all layers of MobileNetV2 base_model_for_build.")
        base_model_for_build.trainable = False

    inputs = Input(shape=input_shape_with_channels, name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x_base = base_model_for_build(x, training=any(layer.trainable for layer in base_model_for_build.layers))

    if 0 < dropout_rate < 1:
        final_features = Dropout(dropout_rate)(x_base)
    else:
        final_features = x_base

    outputs = Dense(num_classes, activation='softmax', name="predictions")(final_features)
    model = Model(inputs=inputs, outputs=outputs, name="CellTypeMobileNetV2")
    return model


def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))
    try:
        plt.subplot(1, 2, 1)
        if 'accuracy' in history.history: plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history: plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy');
        plt.xlabel('Epoch');
        plt.ylabel('Accuracy');
        plt.legend();
        plt.grid(True)
        plt.subplot(1, 2, 2)
        if 'loss' in history.history: plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss');
        plt.xlabel('Epoch');
        plt.ylabel('Loss');
        plt.legend();
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
    except Exception as e:
        print(f"  Warning: Could not generate training plot: {e}")
    finally:
        plt.close()
    print(f"  [PLOT] Saved training plot to {save_path}")


def get_preprocessing_function(cnn_model_name_str: str, target_colorspace_str: str):
    def custom_image_processor(image_np_rgb):
        current_image_float = image_np_rgb.astype(np.float32)
        if np.max(current_image_float) > 1.01:
            current_image_normalized_01 = current_image_float / 255.0
        else:
            current_image_normalized_01 = current_image_float

        if target_colorspace_str == "grayscale":
            gray = rgb2gray(current_image_normalized_01)
            processed_image_intermediate = np.stack((gray,) * 3, axis=-1)
        elif target_colorspace_str == "hed":
            processed_image_intermediate = rgb2hed(current_image_normalized_01)
        elif target_colorspace_str == "ycbcr":
            processed_image_intermediate = rgb2ycbcr(current_image_normalized_01)
        elif target_colorspace_str == "cielab":
            processed_image_intermediate = rgb2lab(current_image_normalized_01)
        elif target_colorspace_str == "original":
            processed_image_intermediate = current_image_normalized_01
        else:
            print(f"Warning: Unknown target_colorspace '{target_colorspace_str}'. Using original RGB.")
            processed_image_intermediate = current_image_normalized_01

        final_output_image_channels = []
        if processed_image_intermediate.ndim == 2:
            rescaled_channel = rescale_intensity(processed_image_intermediate, out_range=(0, 255))
            final_output_image = np.dstack((rescaled_channel, rescaled_channel, rescaled_channel))
        elif processed_image_intermediate.ndim == 3 and processed_image_intermediate.shape[2] == 3:
            for i in range(processed_image_intermediate.shape[2]):
                channel = processed_image_intermediate[:, :, i]
                rescaled_channel = rescale_intensity(channel, out_range=(0, 255))
                final_output_image_channels.append(rescaled_channel)
            final_output_image = np.dstack(final_output_image_channels)
        elif processed_image_intermediate.ndim == 3 and processed_image_intermediate.shape[2] == 1:
            rescaled_channel = rescale_intensity(processed_image_intermediate[:, :, 0], out_range=(0, 255))
            final_output_image = np.dstack((rescaled_channel, rescaled_channel, rescaled_channel))
        else:
            print(
                f"Warning: Unexpected image shape {processed_image_intermediate.shape} after color conversion. Attempting direct scaling from presumed [0,1] to [0,255].")
            final_output_image = (processed_image_intermediate * 255.0).clip(0, 255)

        return final_output_image.astype(np.float32)

    return custom_image_processor


def train_single_fold(fold_num: int, run_cfg: dict, cnn_model_name: str = "mobilenetv2"):
    hparams_all = run_cfg.get('hyperparameters', {})
    hparams_cnn = hparams_all.get(cnn_model_name, {})

    hparams_cnn.setdefault('epochs', hparams_all.get(cnn_model_name, {}).get('epochs', 50))
    hparams_cnn.setdefault('batch_size', hparams_all.get(cnn_model_name, {}).get('batch_size', 32))
    hparams_cnn.setdefault('img_size', hparams_all.get(cnn_model_name, {}).get('img_size', [224, 224]))
    hparams_cnn.setdefault('base_trainable_setting',
                           hparams_all.get(cnn_model_name, {}).get('base_trainable_setting', 15))
    hparams_cnn.setdefault('dropout',
                           hparams_all.get(cnn_model_name, {}).get('dropout', 0.3))
    hparams_cnn.setdefault('learning_rate', hparams_all.get(cnn_model_name, {}).get('learning_rate', 0.0003))
    hparams_cnn.setdefault('optimizer', hparams_all.get(cnn_model_name, {}).get('optimizer', 'adam'))
    hparams_cnn.setdefault('early_stopping_patience',
                           hparams_all.get(cnn_model_name, {}).get('early_stopping_patience', 10))
    hparams_cnn.setdefault('reduce_lr_patience', hparams_all.get(cnn_model_name, {}).get('reduce_lr_patience', 5))

    paths_cfg = run_cfg.get('paths', {})
    cv_cfg = run_cfg.get('cv', {})
    target_colorspace = run_cfg.get('target_colorspace', 'original')

    data_dir = Path(paths_cfg.get('data_dir'))
    splits_dir = Path(paths_cfg.get('splits_dir'))
    output_root_for_exp = Path(paths_cfg.get('output_root'))
    fold_output_dir = output_root_for_exp / f"fold_{fold_num}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n===== Training {cnn_model_name.upper()}: ValFoldNum {fold_num}, TargetColorspace '{target_colorspace}' =====")
    print(f"  Original RGB Data source: {data_dir}")
    print(f"  Output for this run: {fold_output_dir}")
    print(f"  Hyperparameters for {cnn_model_name.upper()}: {hparams_cnn}")

    validation_csv_path = splits_dir / f"fold_{fold_num}.csv"
    if not validation_csv_path.exists():
        print(f"  ERROR: Validation split file {validation_csv_path} not found! Cannot proceed.");
        sys.exit(1)
    df_val_full = pd.read_csv(validation_csv_path)

    k_folds_total = cv_cfg.get('k_folds_total', 5)
    all_possible_fold_indices = range(1, k_folds_total + 1)
    train_fold_indices = [i for i in all_possible_fold_indices if i != fold_num]

    if not train_fold_indices:
        print(
            f"  ERROR: No training fold indices determined (val_fold_num: {fold_num}, total_folds: {k_folds_total}).");
        sys.exit(1)

    print(f"  Loading training data from folds: {train_fold_indices}")
    print(f"  Loading validation data from fold: {fold_num}")

    df_train_list = []
    for train_fold_idx in train_fold_indices:
        train_split_file = splits_dir / f"fold_{train_fold_idx}.csv"
        if train_split_file.exists():
            df_train_list.append(pd.read_csv(train_split_file))
        else:
            print(f"  Warning: Training split file {train_split_file} not found for val_fold {fold_num}.")

    if not df_train_list: print(f"  ERROR: No training data could be loaded for val_fold {fold_num}."); sys.exit(1)
    df_train_full = pd.concat(df_train_list, ignore_index=True)

    current_seed = cv_cfg.get('seed', 42)
    sample_fraction = run_cfg.get('sample_fraction_for_test')
    if sample_fraction and 0 < sample_fraction < 1.0:
        print(f"  [INFO] Test run: Applying sample_fraction {sample_fraction:.2f} to loaded train and validation data.")
        if not df_train_full.empty:
            min_samples_stratify_train = df_train_full['label'].nunique() if 'label' in df_train_full.columns else 1
            if 'label' in df_train_full.columns and df_train_full['label'].nunique() > 1 and len(
                    df_train_full) * sample_fraction >= min_samples_stratify_train:
                try:
                    df_train_full, _ = train_test_split(df_train_full, train_size=sample_fraction,
                                                        stratify=df_train_full['label'], random_state=current_seed)
                except ValueError:
                    df_train_full = df_train_full.sample(frac=sample_fraction, random_state=current_seed)
            else:
                df_train_full = df_train_full.sample(frac=sample_fraction, random_state=current_seed)

        if not df_val_full.empty:
            min_samples_stratify_val = df_val_full['label'].nunique() if 'label' in df_val_full.columns else 1
            if 'label' in df_val_full.columns and df_val_full['label'].nunique() > 1 and len(
                    df_val_full) * sample_fraction >= min_samples_stratify_val:
                try:
                    df_val_full, _ = train_test_split(df_val_full, train_size=sample_fraction,
                                                      stratify=df_val_full['label'], random_state=current_seed)
                except ValueError:
                    df_val_full = df_val_full.sample(frac=sample_fraction, random_state=current_seed)
            else:
                df_val_full = df_val_full.sample(frac=sample_fraction, random_state=current_seed)

    if df_train_full.empty or df_val_full.empty:
        print(
            f"  ERROR: Training data ({len(df_train_full)}) or Validation data ({len(df_val_full)}) is empty after sampling/loading. Skipping.");
        sys.exit(1)

    df_train_full["filepath"] = df_train_full["rel_path"].apply(lambda x: str(data_dir / Path(x)))
    df_val_full["filepath"] = df_val_full["rel_path"].apply(lambda x: str(data_dir / Path(x)))
    df_train_full["label"] = df_train_full["label"].astype(str)
    df_val_full["label"] = df_val_full["label"].astype(str)

    img_size_hw = tuple(hparams_cnn.get('img_size'))
    batch_size = hparams_cnn.get('batch_size')
    epochs = hparams_cnn.get('epochs')

    all_unique_labels = sorted(list(pd.concat([df_train_full['label'], df_val_full['label']]).unique()))
    num_classes = len(all_unique_labels)
    print(f"  Overall {num_classes} classes for this run: {all_unique_labels}")
    if num_classes == 0: print(f"  ERROR: No classes found!"); sys.exit(1)
    if num_classes == 1: print(f"  Warning: Only one class ('{all_unique_labels[0]}') present.")

    image_processor = get_preprocessing_function(cnn_model_name, target_colorspace)

    datagen_train = ImageDataGenerator(
        preprocessing_function=image_processor
        # No rescale here.
    )
    datagen_val = ImageDataGenerator(
        preprocessing_function=image_processor
    )

    print(f"  Training on {len(df_train_full)} images, Validating on {len(df_val_full)} images.")
    train_generator = datagen_train.flow_from_dataframe(
        dataframe=df_train_full, x_col="filepath", y_col="label",
        target_size=img_size_hw, batch_size=batch_size, class_mode="categorical",
        shuffle=True, seed=current_seed, classes=all_unique_labels
    )
    val_generator = datagen_val.flow_from_dataframe(
        dataframe=df_val_full, x_col="filepath", y_col="label",
        target_size=img_size_hw, batch_size=batch_size, class_mode="categorical",
        shuffle=False, classes=all_unique_labels
    )
    if train_generator.n == 0: print(f"  ERROR: Training generator empty."); sys.exit(1)
    if val_generator.n == 0: print(f"  Warning: Validation generator empty.");

    model = build_custom_mobilenetv2_model(
        num_classes=num_classes, img_size_hw=img_size_hw,
        base_trainable_setting=hparams_cnn.get('base_trainable_setting'),
        dropout_rate=hparams_cnn.get('dropout')
    )

    optimizer_choice = hparams_cnn.get('optimizer', 'adam').lower()
    learning_rate = hparams_cnn.get('learning_rate')
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        try:
            OptimizerClass = getattr(tf.keras.optimizers, optimizer_choice.capitalize());
            optimizer = OptimizerClass(
                learning_rate=learning_rate)
        except AttributeError:
            print(f"  Warning: Optimizer '{optimizer_choice}' not found. Using Adam.");
            optimizer = Adam(
                learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"  Model Summary ({cnn_model_name.upper()}):")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    ckpt_path_str = str(fold_output_dir / "model_best.keras")
    print(f"  Best model will be saved to: {ckpt_path_str}")
    callbacks = [
        ModelCheckpoint(ckpt_path_str, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=hparams_cnn.get('early_stopping_patience', 10),
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=hparams_cnn.get('reduce_lr_patience', 5),
                          min_lr=1e-6, verbose=1)
    ]

    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator if val_generator.n > 0 else None,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time();
    training_duration = end_time - start_time
    (fold_output_dir / "training_time.txt").write_text(f"{training_duration:.2f} seconds")
    print(f"  Training ({cnn_model_name.upper()}) completed in {training_duration:.2f} seconds.")

    model.save(str(fold_output_dir / "model_final.keras"))

    if Path(ckpt_path_str).exists():
        best_model = tf.keras.models.load_model(ckpt_path_str, compile=False)
    else:
        print(
            f"  Warning: Best model at {ckpt_path_str} not found. Using final model for evaluation.");
        best_model = model

    if val_generator.n > 0:
        y_pred_probs = best_model.predict(val_generator, verbose=0)
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        y_true_indices = val_generator.classes
        idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
        report_target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        report_str = classification_report(y_true_indices, y_pred_indices, target_names=report_target_names, digits=4,
                                           zero_division=0)
        report_dict = classification_report(y_true_indices, y_pred_indices, target_names=report_target_names, digits=4,
                                            output_dict=True, zero_division=0)
        print(
            f"\n  Classification Report ({cnn_model_name.upper()}, ValFoldNum {fold_num}, TargetColorspace '{target_colorspace}'):\n{report_str}")
        (fold_output_dir / "report.txt").write_text(report_str)
        with open(fold_output_dir / "report_dict.json", "w", encoding='utf-8') as f:
            json.dump(report_dict, f, indent=4)
        np.save(str(fold_output_dir / "y_true.npy"), y_true_indices)
        np.save(str(fold_output_dir / "y_pred.npy"), y_pred_indices)
        np.save(str(fold_output_dir / "y_pred_probs.npy"), y_pred_probs)
        cm_labels_indices = [val_generator.class_indices[name] for name in report_target_names]
        cm = confusion_matrix(y_true_indices, y_pred_indices, labels=cm_labels_indices)
        np.savetxt(str(fold_output_dir / "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    else:
        print(f"  Skipping final evaluation metrics as validation generator was empty or not used.")
        (fold_output_dir / "report.txt").write_text("Evaluation skipped due to empty/unused validation set.")

    plot_training_history(history, fold_output_dir / "training_performance.png")
    print(
        f"  [OK] Run for ValFoldNum {fold_num} (TargetColorspace '{target_colorspace}') processing finished. Results in {fold_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 model for a specific fold/run configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the run-specific YAML configuration file.")
    args = parser.parse_args()

    try:
        run_config = load_run_config(Path(args.config))
    except FileNotFoundError:
        print(f"FATAL: Run config file not found at {args.config}");
        sys.exit(1)
    except yaml.YAMLError:
        print(f"FATAL: Error parsing YAML run config at {args.config}");
        sys.exit(1)

    fold_to_validate_on = run_config.get('cv', {}).get('current_fold_num_to_run')
    if fold_to_validate_on is None: print("FATAL ERROR: 'current_fold_num_to_run' not in CV config."); sys.exit(1)

    try:
        train_single_fold(fold_num=fold_to_validate_on, run_cfg=run_config, cnn_model_name="mobilenetv2")
    except Exception as e:
        print(f"FATAL ERROR during train_single_fold for mobilenetv2, ValFoldNum {fold_to_validate_on}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)