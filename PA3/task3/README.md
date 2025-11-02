# Advanced Topics in Machine Learning - Assignment 3: Task 3 (Knowledge Distillation)

This repository contains the complete implementation and analysis for Task 3 of the assignment, focusing on Knowledge Distillation. All experiments are designed to be modular, reproducible, and self-contained.

## Project Structure and File Guide

The project is organized into core modules, main training scripts, and analysis scripts. Here is a guide to what each file does and which sub-task it corresponds to.

### 1. Core Modules (The Framework)

These files form the backbone of all experiments and are used across multiple tasks.

-   `configs.py`: A centralized file for all hyperparameters (learning rates, epochs, distillation parameters, etc.).
-   `data_loader.py`: Handles loading and transforming the CIFAR-100 dataset, including standard and color-jittered versions.
-   `losses.py`: Contains the implementation of all custom distillation loss functions (`DistillationLoss`, `DKDLoss`, `CRDLoss`).
-   `models_with_embeddings.py`: A PyTorch wrapper to extract pre-classifier embeddings from VGG models, used for CRD.
-   `models_with_hints.py`: A PyTorch wrapper to extract intermediate feature maps from VGG models, used for FitNets.
-   `trainers.py`: Contains the generic `train_epoch` and `evaluate` functions used for baseline training and all model evaluations.
-   `trainers_kd.py`: Contains specialized training loops for various distillation methods (`train_epoch_kd`, `train_epoch_crd`).
-   `utils.py`: General helper functions for saving checkpoints and calculating accuracy.
-   `prepare_raw_data.py`: A one-time utility to extract raw CIFAR-100 images for clean Grad-CAM visualizations.

---

### 2. Task 3.1 & 3.2: Training the Models

These are the main executable scripts used to train each of the student models.

-   `evaluate_teacher.py`: Establishes the performance ceiling of the pre-trained VGG-16 teacher.
-   `main_baseline.py`: Trains the independent VGG-11 student (`S_I`) as a baseline.
-   `main_label_smoothing.py`: Trains the student with Label Smoothing Regularization (LSR).
-   `main_kd_lm.py`: Trains a student using standard **Logit Matching (LM)**.
-   `main_dkd.py`: Trains a student using **Decoupled Knowledge Distillation (DKD)**.
-   `main_crd.py`: Trains a student using **Contrastive Representation Distillation (CRD)**.
-   `main_fitnets_part1.py` & `main_fitnets_part2.py`: The two-stage training process for the **FitNets (Hint-based)** student.

---

### 3. Task 3.3: Analysis of Probability Distributions

-   `analysis_kl_divergence.py`: Loads all trained models and calculates the average KL Divergence between each student's and the teacher's output distributions on the test set.

---

### 4. Task 3.4: Analysis of Localization Knowledge (Grad-CAM)

This analysis is split into qualitative (visual) and quantitative (cosine similarity) parts.

-   `analysis_gradcam_visual.py`: Generates the main 4x7 grid of Grad-CAM heatmaps for visual comparison in the report.
-   `discover_hard_set.py`: A utility script to identify and save the indices of "hard" images (where the teacher is correct but the baseline fails).
-   `discover_easy_set.py`: A utility script to identify and save the indices of "easy" images (where both teacher and baseline are correct).
-   `analysis_gradcam_quant.py`: Performs the quantitative cosine similarity comparison across the **full** 10,000-image test set.
-   `analysis_gradcam_quant_hard.py`: Runs the same analysis exclusively on the pre-computed **"hard set"**.
-   `analysis_gradcam_quant_easy.py`: Runs the same analysis exclusively on the pre-computed **"easy set"**.

---

### 5. Task 3.5: Analysis of Color Invariance Transfer

This multi-stage experiment tests the distillation of an abstract property.

-   `finetune_teacher_color.py`: **(Step 1)** Fine-tunes the VGG-16 teacher with aggressive color jitter to create a "color-invariant" expert.
-   `main_distill_from_color_inv.py`: **(Step 2)** A unified script to distill knowledge from the color-invariant teacher to new students using LM, DKD, or CRD.
-   `main_fitnets_color_inv.py`: **(Step 2 for FitNets)** The two-stage training for FitNets using the color-invariant teacher.
-   `evaluate_color_invariance.py`: **(Step 3)** The final evaluation script. It loads all original and all color-distilled students and compares their performance on a color-jittered test set.

---

### 6. Task 3.6: Analysis of a Larger Teacher

This experiment tests the impact of teacher model size on student performance.

-   `main_kd_lm_vgg19.py`: Trains a student using Logit Matching, but with a larger, pre-trained VGG-19 as the teacher.