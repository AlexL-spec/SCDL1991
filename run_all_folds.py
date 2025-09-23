# src/run_all_folds.py
import yaml
from pathlib import Path
import subprocess
import pandas as pd
import datetime
import shutil
import sys
import os
import argparse
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent


def load_config(config_path_from_root="config.yaml"):
    config_file = PROJECT_ROOT / config_path_from_root
    if not config_file.exists():
        print(f"FATAL ERROR: Main configuration file not found at {config_file}")
        sys.exit(1)
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_temporary_split_csv(original_split_csv_path: Path,
                               temp_target_dir: Path,
                               sample_fraction: float,
                               seed: int,
                               fold_num_for_log: int,
                               global_splits_dir_path: Path) -> Path:
    temp_target_dir.mkdir(parents=True, exist_ok=True)

    if not original_split_csv_path.exists():
        print(
            f"      Warning: Original split CSV {original_split_csv_path} for fold {fold_num_for_log} not found. Cannot create temp split.")
        temp_csv_path_missing = temp_target_dir / f"{original_split_csv_path.stem}_testfrac_missing_original.csv"
        headers = ['rel_path', 'label']
        try:
            any_other_fold = next(global_splits_dir_path.glob("fold_*.csv"), None)
            if any_other_fold and any_other_fold.exists():
                headers = pd.read_csv(any_other_fold, nrows=0).columns.tolist()
            else:
                print(
                    f"      Warning: Could not find other fold CSVs in {global_splits_dir_path} to determine headers for missing {original_split_csv_path.name}. Using default headers.")
        except Exception as e_header:
            print(
                f"      Warning: Error trying to get headers from other folds for {original_split_csv_path.name}: {e_header}. Using default headers.")
        pd.DataFrame(columns=headers).to_csv(temp_csv_path_missing, index=False, encoding='utf-8')
        return temp_csv_path_missing

    df_orig_split = pd.read_csv(original_split_csv_path)
    if df_orig_split.empty:
        print(f"      Warning: Original split CSV {original_split_csv_path} for fold {fold_num_for_log} is empty.")
        temp_csv_path_empty = temp_target_dir / f"{original_split_csv_path.stem}_testfrac_empty.csv"
        df_orig_split.head(0).to_csv(temp_csv_path_empty, index=False, encoding='utf-8')
        return temp_csv_path_empty

    num_to_sample_ideal = int(len(df_orig_split) * sample_fraction)
    if num_to_sample_ideal == 0 and len(df_orig_split) > 0:
        num_to_sample_ideal = 1

    df_test_split = pd.DataFrame()
    if 'label' in df_orig_split.columns and df_orig_split['label'].nunique() > 1 and len(df_orig_split) > df_orig_split[
        'label'].nunique() and num_to_sample_ideal > 0:
        try:
            min_samples_for_stratify = df_orig_split['label'].nunique()
            actual_sample_size = max(num_to_sample_ideal,
                                     min_samples_for_stratify if num_to_sample_ideal >= min_samples_for_stratify else 0)
            actual_sample_size = min(actual_sample_size, len(df_orig_split))

            if actual_sample_size < min_samples_for_stratify and actual_sample_size > 0:
                print(
                    f"      Warning: Cannot stratify sample for fold {fold_num_for_log} with {actual_sample_size} samples and {min_samples_for_stratify} classes. Using random sample.")
                df_test_split = df_orig_split.sample(n=actual_sample_size, random_state=seed, replace=False if len(
                    df_orig_split) >= actual_sample_size else True)
            elif actual_sample_size > 0:
                df_test_split, _ = train_test_split(
                    df_orig_split, train_size=actual_sample_size, stratify=df_orig_split['label'], random_state=seed
                )
            if df_test_split.empty and not df_orig_split.empty and num_to_sample_ideal > 0:
                df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed)
        except ValueError as e:
            print(
                f"      Warning: Stratified sampling failed for fold {fold_num_for_log} (sample_fraction: {sample_fraction}, error: {e}). Falling back to random sample.")
            df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed,
                                                 replace=False if len(df_orig_split) >= min(num_to_sample_ideal,
                                                                                            len(df_orig_split)) else True)
    elif not df_orig_split.empty and num_to_sample_ideal > 0:
        df_test_split = df_orig_split.sample(n=min(num_to_sample_ideal, len(df_orig_split)), random_state=seed,
                                             replace=False if len(df_orig_split) >= min(num_to_sample_ideal,
                                                                                        len(df_orig_split)) else True)

    if df_test_split.empty and not df_orig_split.empty and num_to_sample_ideal > 0:
        df_test_split = df_orig_split.sample(n=1, random_state=seed)
    temp_csv_path = temp_target_dir / f"{original_split_csv_path.stem}_samplefrac_{sample_fraction:.3f}.csv"
    df_test_split.to_csv(temp_csv_path, index=False, encoding='utf-8')
    return temp_csv_path


def run_command(command_list: list, step_name: str, error_on_fail=True, timeout_seconds=7200,
                pbar: tqdm = None):
    str_command_list = [str(c) for c in command_list]

    original_postfix = ""
    if pbar and hasattr(pbar, 'postfix') and pbar.postfix is not None:
        original_postfix = str(pbar.postfix)

    current_action_postfix = f"Running: {step_name[:30]}..."
    if pbar:
        pbar.set_postfix_str(
            f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].split(', TIMEOUT:')[0].split(', ERROR:')[0].split(', NOT FOUND:')[0].split(', EXCEPTION:')[0].strip()}, {current_action_postfix}")
    else:
        print(f"--- Starting: {step_name} ---")
        print(f"      Command: {' '.join(str_command_list)}")

    process = None
    try:
        process = subprocess.Popen(str_command_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   encoding='utf-8',
                                   errors='replace',
                                   bufsize=1,
                                   universal_newlines=True)

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    tqdm.write(f"      [LOG - {step_name[:20]}]: {line}", file=sys.stdout)
                    if pbar and (line.startswith(
                            "Epoch ") or "val_loss" in line or "Report" in line or "summary written" in line):
                        pbar.set_postfix_str(
                            f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, Last: {line[:50]}...")

        process.wait(timeout=timeout_seconds)
        stderr_output = process.stderr.read() if process.stderr else ""

        if process.returncode != 0:
            final_postfix_msg = f"ERROR: {step_name[:30]}"
            if pbar: pbar.set_postfix_str(
                f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, {final_postfix_msg}")
            print(f"ERROR during {step_name} (return code: {process.returncode}): {' '.join(str_command_list)}")
            if stderr_output.strip():
                print(f"      Full Stderr from failed command:\n{stderr_output.strip()}")
            if error_on_fail:
                raise subprocess.CalledProcessError(process.returncode, str_command_list,
                                                    stderr=stderr_output)
            return False
        else:
            final_postfix_msg = f"Finished: {step_name[:30]}"
            if pbar:  pbar.set_postfix_str(
                f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, {final_postfix_msg}")
            if stderr_output.strip():
                tqdm.write(f"      [STDERR - {step_name[:20]} - Non-fatal]:\n{stderr_output.strip()}", file=sys.stderr)
            return True

    except subprocess.TimeoutExpired:
        final_postfix_msg = f"TIMEOUT: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(
            f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, {final_postfix_msg}")
        print(f"TIMEOUT ERROR: {step_name} exceeded {timeout_seconds}s. Command: {' '.join(str_command_list)}")
        if process:
            process.kill()
            try:
                stdout_after_timeout, stderr_after_timeout = process.communicate(timeout=5)
                if stdout_after_timeout and stdout_after_timeout.strip():
                    tqdm.write(f"      [LOG after timeout - {step_name[:20]}]: {stdout_after_timeout.strip()}",
                               file=sys.stdout)
                if stderr_after_timeout and stderr_after_timeout.strip():
                    tqdm.write(f"      [STDERR after timeout - {step_name[:20]}]: {stderr_after_timeout.strip()}",
                               file=sys.stderr)
            except Exception as e_comm:
                print(f"      Error communicating with timed-out process: {e_comm}")
        if error_on_fail: raise
        return False
    except FileNotFoundError as e:
        final_postfix_msg = f"NOT FOUND: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(
            f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, {final_postfix_msg}")
        print(
            f"ERROR: Command or script not found for {step_name}. Command: {' '.join(str_command_list)}. Details: {e}")
        if error_on_fail: raise
        return False
    except Exception as e:
        final_postfix_msg = f"EXCEPTION: {step_name[:30]}"
        if pbar: pbar.set_postfix_str(
            f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}, {final_postfix_msg}")
        print(f"AN UNEXPECTED ERROR OCCURRED during {step_name}: {e}")
        if process:
            process.kill()
            try:
                _, stderr_on_exception = process.communicate(timeout=5)
                if stderr_on_exception and stderr_on_exception.strip():
                    tqdm.write(f"      [STDERR on exception - {step_name[:20]}]: {stderr_on_exception.strip()}",
                               file=sys.stderr)
            except Exception as e_comm_ex:
                print(f"      Error communicating with process on exception: {e_comm_ex}")
        if error_on_fail: raise
        return False
    finally:
        if pbar:
            pbar.set_postfix_str(
                f"{original_postfix.split(', Running:')[0].split(', Last:')[0].split(', Finished:')[0].strip()}")


def main(run_mode: str):
    cfg = load_config()
    python_executable = sys.executable
    is_test_run = (run_mode == "test")

    print(f"--- Starting Experiment Orchestration (Mode: {run_mode.upper()}) ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python Executable: {python_executable}")

    current_run_timeout = cfg.get('subprocess_timeout_seconds', 36000)
    if is_test_run and cfg.get('test_run', {}).get('subprocess_timeout_seconds_override') is not None:
        current_run_timeout = cfg['test_run']['subprocess_timeout_seconds_override']
        print(f"Using TEST MODE timeout: {current_run_timeout} seconds")
    else:
        print(f"Using FULL MODE timeout: {current_run_timeout} seconds")

    folds_to_process_numbers = list(range(1, cfg['cv'].get('k_folds', 5) + 1))
    test_cfg = cfg.get('test_run', {})
    if is_test_run and test_cfg.get('folds_to_run_override'):
        folds_to_process_numbers_override = test_cfg['folds_to_run_override']
        if isinstance(folds_to_process_numbers_override, list) and folds_to_process_numbers_override:
            folds_to_process_numbers = folds_to_process_numbers_override
        else:
            print("Warning: test_run.folds_to_run_override is invalid or empty. Defaulting to [1] for test safety.")
            folds_to_process_numbers = [1]

    active_hyperparameters = cfg['hyperparameters'].copy()
    sample_fraction_for_test_run = 1.0

    if is_test_run:
        sample_fraction_for_test_run = test_cfg.get('sample_fraction', 0.1)
        print_test_overrides = {}
        for model_key in cfg['experiments']['model_types']:
            if model_key in active_hyperparameters:
                if model_key in ["resnet", "mobilenetv2", "efficientnetb0"]:
                    if f"{model_key}_epochs_override" in test_cfg:
                        active_hyperparameters[model_key]['epochs'] = test_cfg[f"{model_key}_epochs_override"]
                        print_test_overrides[f"{model_key.capitalize()}-Epochs"] = active_hyperparameters[model_key][
                            'epochs']
                if model_key == "rf":
                    if "rf_n_estimators_override" in test_cfg:
                        active_hyperparameters['rf']['n_estimators'] = test_cfg['rf_n_estimators_override']
                        print_test_overrides["RF-N_est"] = active_hyperparameters['rf']['n_estimators']
        print(
            f"!!! TEST MODE ENABLED: Folds to process (for validation)= {folds_to_process_numbers}, SampleFrac for CSVs={sample_fraction_for_test_run:.3f}, Overrides: {print_test_overrides} !!!")
    else:
        print(
            f"--- FULL MODE ENABLED: Folds to process (for validation)={folds_to_process_numbers}, Using full samples from CSVs and hyperparameters from config. ---")

    temp_sampled_splits_dir = PROJECT_ROOT / cfg['paths']['splits_dir'] / "temp_orchestrator_sampled_splits"

    global_splits_dir = PROJECT_ROOT / cfg['paths']['splits_dir']
    hog_output_file_stems_for_kfold_setup = {}
    all_available_hog_stems_json_str = "[]"
    ml_models_requiring_hog = [mt for mt in cfg['experiments']['model_types'] if mt in ["rf", "svm"]]

    temporary_sampling_directory_needed = False

    if is_test_run and sample_fraction_for_test_run < 1.0:
        temporary_sampling_directory_needed = True
    elif not is_test_run and "rf" in cfg['experiments']['model_types']:
        rf_hp = cfg.get('hyperparameters', {}).get('rf', {})
        if rf_hp.get('full_mode_sample_fraction') is not None and rf_hp['full_mode_sample_fraction'] < 1.0:
            temporary_sampling_directory_needed = True

    if temporary_sampling_directory_needed:
        if temp_sampled_splits_dir.exists(): shutil.rmtree(temp_sampled_splits_dir)
        temp_sampled_splits_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporary sampled splits will be stored in: {temp_sampled_splits_dir}")

    if ml_models_requiring_hog:
        print(f"ML models requiring HOG features detected: {ml_models_requiring_hog}. Preparing HOG features...")
        all_k_fold_indices = list(range(1, cfg['cv'].get('k_folds', 5) + 1))
        print(f"    Determining HOG output file stems for K-Fold setup (up to {cfg['cv'].get('k_folds', 5)} folds)...")

        for fold_idx_k_orig in all_k_fold_indices:
            original_fold_csv_stem = f"fold_{fold_idx_k_orig}"
            original_split_csv_path = global_splits_dir / f"{original_fold_csv_stem}.csv"

            current_hog_input_sample_fraction = 1.0
            use_temp_csv_for_hog_input = False
            target_temp_dir_for_csv = temp_sampled_splits_dir

            if is_test_run and sample_fraction_for_test_run < 1.0:
                current_hog_input_sample_fraction = sample_fraction_for_test_run
                use_temp_csv_for_hog_input = True
            elif not is_test_run:
                rf_config = cfg.get('hyperparameters', {}).get('rf', {})
                rf_full_mode_fraction = rf_config.get('full_mode_sample_fraction')
                if "rf" in cfg['experiments'][
                    'model_types'] and rf_full_mode_fraction is not None and rf_full_mode_fraction < 1.0:
                    current_hog_input_sample_fraction = rf_full_mode_fraction
                    use_temp_csv_for_hog_input = True

            if use_temp_csv_for_hog_input:

                temp_csv_path_for_hog_input = create_temporary_split_csv(
                    original_split_csv_path,
                    target_temp_dir_for_csv,
                    current_hog_input_sample_fraction,
                    cfg['cv']['seed'],
                    fold_idx_k_orig,
                    global_splits_dir_path=global_splits_dir
                )
                hog_output_file_stems_for_kfold_setup[fold_idx_k_orig] = temp_csv_path_for_hog_input.stem
            else:
                hog_output_file_stems_for_kfold_setup[fold_idx_k_orig] = original_fold_csv_stem

        print(
            f"      HOG input CSV stems that will be used/generated: {list(hog_output_file_stems_for_kfold_setup.values())}")
        all_available_hog_stems_for_kfold_models = list(hog_output_file_stems_for_kfold_setup.values())
        all_available_hog_stems_json_str = json.dumps(all_available_hog_stems_for_kfold_models)

        print(f"    Ensuring HOG features are ready for applicable colorspaces for K-Fold setup...")
        base_feature_dir_path_str = cfg['paths'].get('base_feature_dir')
        if not base_feature_dir_path_str:
            print(
                f"      FATAL ERROR: 'paths.base_feature_dir' is not defined in config.yaml, but RF/SVM models are specified. Cannot proceed with HOG.")
            sys.exit(1)
        base_feature_dir_path = PROJECT_ROOT / base_feature_dir_path_str

        num_hog_tasks = len(cfg['experiments']['colorspaces']) * len(all_k_fold_indices)
        with tqdm(total=num_hog_tasks, desc="HOG Feature Extraction", unit="task") as hog_pbar:
            for colorspace_for_hog in cfg['experiments']['colorspaces']:
                for original_fold_num_for_hog_extraction in all_k_fold_indices:
                    hog_input_csv_stem = hog_output_file_stems_for_kfold_setup.get(original_fold_num_for_hog_extraction)
                    if not hog_input_csv_stem:
                        hog_pbar.update(1)
                        continue

                    splits_dir_for_hog_command = global_splits_dir
                    is_hog_input_sampled = False
                    if is_test_run and sample_fraction_for_test_run < 1.0:
                        is_hog_input_sampled = True
                    elif not is_test_run:
                        rf_config = cfg.get('hyperparameters', {}).get('rf', {})
                        rf_full_mode_fraction = rf_config.get('full_mode_sample_fraction')
                        if "rf" in cfg['experiments'][
                            'model_types'] and rf_full_mode_fraction is not None and rf_full_mode_fraction < 1.0:
                            is_hog_input_sampled = True

                    if is_hog_input_sampled:
                        splits_dir_for_hog_command = temp_sampled_splits_dir

                    hog_command = [
                        python_executable, str(SRC_ROOT / "ml/feature_extractor/hog.py"),
                        "--image_data_root", str(PROJECT_ROOT / cfg['paths']['base_data_dir']),
                        "--colorspace", colorspace_for_hog,
                        "--splits_dir", str(splits_dir_for_hog_command),
                        "--fold_id", hog_input_csv_stem,
                        "--output_dir_base", str(base_feature_dir_path)
                    ]
                    expected_hog_file = base_feature_dir_path / f"hog_{colorspace_for_hog}" / f"{hog_input_csv_stem}_hog_features.pkl"

                    hog_pbar.set_description(f"HOG ({colorspace_for_hog}, CSV: {hog_input_csv_stem[:25]})")
                    if expected_hog_file.exists():
                        tqdm.write(
                            f"      HOG features for ({colorspace_for_hog}, CSV Stem: {hog_input_csv_stem}) already exist. Skipping extraction.",
                            file=sys.stdout)
                    else:
                        run_command(hog_command, f"HOG Extr ({colorspace_for_hog}, {hog_input_csv_stem[:15]})",
                                    timeout_seconds=current_run_timeout, pbar=None)
                    hog_pbar.update(1)
    else:
        print("No models requiring HOG features (e.g., rf, svm) are listed in experiments. Skipping HOG preparation.")

    num_colorspaces = len(cfg['experiments']['colorspaces'])
    num_model_types = len(cfg['experiments']['model_types'])
    num_validation_folds_to_process = len(folds_to_process_numbers)
    total_iterations = num_colorspaces * num_model_types * num_validation_folds_to_process

    overall_pbar_desc = "Full Experiment Run" if not is_test_run else "Test Experiment Run"
    experiment_pbar = tqdm(total=total_iterations, desc=overall_pbar_desc, unit="task", position=0, leave=True)

    for colorspace in cfg['experiments']['colorspaces']:
        current_colorspace_data_dir = PROJECT_ROOT / cfg['paths']['base_data_dir'] / colorspace
        if not current_colorspace_data_dir.exists():
            tqdm.write(
                f"SKIPPING Colorspace: Data directory for colorspace {colorspace} not found: {current_colorspace_data_dir}.",
                file=sys.stderr)
            experiment_pbar.update(num_model_types * num_validation_folds_to_process)
            continue

        for model_type in cfg['experiments']['model_types']:
            current_model_hparams_for_run_cfg = active_hyperparameters.get(model_type, {})
            if not current_model_hparams_for_run_cfg and model_type in cfg['experiments']['model_types']:
                tqdm.write(
                    f"    WARNING: Hyperparameters for model_type '{model_type}' not found in config, but it's in active model_types. Skipping this model for {colorspace}.",
                    file=sys.stderr)
                experiment_pbar.update(num_validation_folds_to_process)
                continue

            timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

            exp_name_suffix_parts = []
            if is_test_run:
                exp_name_suffix_parts.append(f"testrun_frac{sample_fraction_for_test_run:.2f}")
            else:
                exp_name_suffix_parts.append("fullrun")
                if model_type == "rf":
                    rf_hp = cfg.get('hyperparameters', {}).get('rf', {})
                    rf_fm_frac = rf_hp.get('full_mode_sample_fraction')
                    if rf_fm_frac is not None and rf_fm_frac < 1.0:
                        exp_name_suffix_parts.append(f"rf_frac{rf_fm_frac:.2f}")

            exp_name_suffix = "_".join(exp_name_suffix_parts) if exp_name_suffix_parts else ""

            exp_name = f"{colorspace}_{model_type}_exp_{timestamp}_{exp_name_suffix}"
            current_experiment_output_dir = PROJECT_ROOT / cfg['paths']['base_output_dir'] / model_type / exp_name
            current_experiment_output_dir.mkdir(parents=True, exist_ok=True)

            experiment_pbar.set_description(f"{overall_pbar_desc} (CS: {colorspace}, Model: {model_type})")

            for val_fold_num in folds_to_process_numbers:
                experiment_pbar.set_postfix_str(f"ValFold: {val_fold_num}/{cfg['cv']['k_folds']}")

                fold_id_str_for_output = f"fold_{val_fold_num}"
                training_cmd_args = []
                known_cnn_models = ["resnet", "mobilenetv2", "efficientnetb0"]

                if model_type in known_cnn_models:
                    cnn_splits_dir = str(global_splits_dir)

                    run_specific_cnn_cfg = {
                        'hyperparameters': {model_type: current_model_hparams_for_run_cfg},
                        'paths': {'data_dir': str(current_colorspace_data_dir),
                                  'splits_dir': cnn_splits_dir,
                                  'output_root': str(current_experiment_output_dir)},
                        'cv': {'k_folds_total': cfg['cv']['k_folds'],
                               'current_fold_num_to_run': val_fold_num,
                               'seed': cfg['cv']['seed']}
                    }
                    if is_test_run and sample_fraction_for_test_run < 1.0:
                        run_specific_cnn_cfg['sample_fraction_for_test'] = sample_fraction_for_test_run

                    temp_run_config_path = current_experiment_output_dir / f"run_cfg_{model_type}_{fold_id_str_for_output}.yaml"
                    with open(temp_run_config_path, 'w', encoding='utf-8') as f_cfg:
                        yaml.dump(run_specific_cnn_cfg, f_cfg)

                    train_script_path = SRC_ROOT / "dl" / model_type / "train.py"
                    if not train_script_path.exists():
                        tqdm.write(
                            f"    ERROR: Training script not found at {train_script_path} for model {model_type}. Skipping.",
                            file=sys.stderr)
                        experiment_pbar.update(1)
                        continue
                    training_cmd_args = [python_executable, str(train_script_path), "--config",
                                         str(temp_run_config_path)]

                elif model_type in ml_models_requiring_hog:
                    if not ml_models_requiring_hog:
                        tqdm.write(
                            f"    Skipping {model_type} as HOG features were not prepared (no HOG-dependent models configured).",
                            file=sys.stderr)
                        experiment_pbar.update(1)
                        continue

                    base_feature_dir_for_ml = PROJECT_ROOT / cfg['paths']['base_feature_dir']
                    if not base_feature_dir_for_ml.exists():
                        tqdm.write(
                            f"    ERROR: 'paths.base_feature_dir' ({base_feature_dir_for_ml}) not found, required for {model_type}. Skipping.",
                            file=sys.stderr)
                        experiment_pbar.update(1)
                        continue

                    current_val_hog_stem = hog_output_file_stems_for_kfold_setup.get(val_fold_num)
                    if not current_val_hog_stem:
                        tqdm.write(
                            f"    SKIPPING {model_type} training for K-Fold (Val Fold Num: {val_fold_num}) - HOG stem for validation not found.",
                            file=sys.stderr)
                        experiment_pbar.update(1)
                        continue

                    fold_specific_output_dir_ml = current_experiment_output_dir / fold_id_str_for_output

                    common_ml_args = [
                        "--output_dir", str(fold_specific_output_dir_ml),
                        "--base_feature_dir", str(base_feature_dir_for_ml),
                        "--colorspace", colorspace,
                        "--current_val_fold_id_stem", current_val_hog_stem,
                        "--all_fold_id_stems_json", all_available_hog_stems_json_str,
                        "--random_state", str(active_hyperparameters.get(model_type, {}).get('random_state', 42))
                    ]
                    if model_type == "rf":
                        rf_hparams = active_hyperparameters.get("rf", {})
                        training_cmd_args = [python_executable, str(SRC_ROOT / "ml/rf/train.py")] + common_ml_args + [
                            "--n_estimators", str(rf_hparams.get('n_estimators', 100)),
                            "--min_samples_split", str(rf_hparams.get('min_samples_split', 2)),
                            "--min_samples_leaf", str(rf_hparams.get('min_samples_leaf', 1))
                        ]
                        if rf_hparams.get('max_depth') is not None:
                            training_cmd_args.extend(["--max_depth", str(rf_hparams.get('max_depth'))])
                    else:
                        tqdm.write(
                            f"    Logic for ML model {model_type} not fully implemented in orchestrator. Skipping.",
                            file=sys.stderr)
                        experiment_pbar.update(1);
                        continue

                else:
                    tqdm.write(f"    Skipping model type '{model_type}' - no specific training logic configured.",
                               file=sys.stderr)
                    experiment_pbar.update(1)
                    continue

                if training_cmd_args:
                    run_command(training_cmd_args,
                                f"Training ({model_type}, {colorspace}, Val Fold: {val_fold_num})",
                                timeout_seconds=current_run_timeout,
                                pbar=experiment_pbar)
                else:
                    tqdm.write(
                        f"    No training command generated for {model_type}, {colorspace}, Val Fold: {val_fold_num}. Skipping.",
                        file=sys.stderr)

                experiment_pbar.update(1)

    experiment_pbar.set_description(f"{overall_pbar_desc} (COMPLETED)")
    experiment_pbar.set_postfix_str("All tasks complete!")
    experiment_pbar.close()

    if temporary_sampling_directory_needed and temp_sampled_splits_dir.exists():
        print(f"\nCleaning up temporary sampled split directory: {temp_sampled_splits_dir}")
        try:
            shutil.rmtree(temp_sampled_splits_dir)
        except OSError as e:
            print(f"Warning: Could not remove temporary directory {temp_sampled_splits_dir}: {e}")

    print("\n--- All Experiments Orchestration Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments for Image Analysis.")
    parser.add_argument("--mode", type=str, default="full", choices=["test", "full"],
                        help="Run mode: 'test' for quick debug runs with subset of data/params, 'full' for complete training.")
    cli_args = parser.parse_args()
    main(run_mode=cli_args.mode)