# 🌊 VaayuChakshu: Physics-Aware Flood Mapping using Multi-Channel cGANs

> A deep learning-based flood monitoring pipeline that integrates physics-inspired loss functions, water index constraints, and structured generation. Trained on real-world Sentinel-1 (Radar) and Sentinel-2 (Optical) satellite pairs.

---

## 📊 Project Overview
This project proposes a **physics-aware SAR-to-Optical generation framework for flood mapping**. The objective is not just image translation (making radar look pretty), but domain-specific flood analysis. We move beyond traditional Generative Adversarial Networks (GANs) by adding meaningful **hydrological supervision**, **cloud-aware learning**, and **physically-consistent representations** to generate accurate flood masks from raw radar data.

### ✨ Highlights:
* **Water-Aware Generation**: The model outputs both a colorful RGB image and a binary Water Flood Mask directly.
* **NDWI Consistency**: Reinforces spectral water behavior in RGB bands based on real physics.
* **Cloud-Aware Losses**: Loss functions are masked using cloud masks to ignore cloudy pixels.
* **Custom Generator/Discriminator**: Modified U-Net + PatchGAN.
* **Speckle-Aware Texture Loss**: Preserves fine SAR radar edges in the generated RGB output.

---

## 📖 Key Terminologies
If you are new to Remote Sensing or Deep Learning, here are the core concepts used in this project:
* **SAR (Synthetic Aperture Radar):** Radar imaging (like Sentinel-1) that bounces radio waves off the earth. It can "see" through clouds, rain, and the dark of night, making it perfect for monitoring floods during storms.
* **Optical Imagery:** Standard visual satellite images (like Sentinel-2 or Google Earth). These are easy for humans to understand but are entirely blocked by heavy clouds.
* **cGAN (Conditional GAN):** A type of AI that pits two neural networks against each other (a Generator and a Discriminator). It translates an input image (SAR) into an output image (Optical/Flood Mask).
* **NDWI (Normalized Difference Water Index):** A standard remote sensing formula used to highlight surface water in satellite images.
* **Mode Collapse:** A common failure in GANs where the AI starts producing the exact same static noise for every input. We solved this using a custom Physics Loss!

---

## 🚀 Quick Start: Run Inference (No Training Required)

If you just want to use the trained AI to detect floods on your own radar images, follow these steps:

**1. Clone the Code & Install Dependencies**
```bash
git clone https://github.com/Weptsugar/VaayuChakshu.git
cd VaayuChakshu
pip install -r requirements.txt
```

**2. Download the Trained Model Weights**
Download our pre-trained model checkpoint (`model_final.ckpt`) from the Releases tab (or ask the repository owner for the link) and save it to your computer.

**3. Run Inference**
Feed the AI a raw Sentinel-1 VV and VH radar image, and it will generate the flood mask!
```bash
python inference.py \
  --checkpoint "path/to/model_final.ckpt" \
  --s1_vv "path/to/VV.tif" \
  --s1_vh "path/to/VH.tif" \
  --output_dir "outputs/results"
```

---

## 🏋️‍♂️ Training from Scratch (Using Kaggle)

Because training this model requires a powerful GPU and hours of processing time, we highly recommend using Kaggle's free GPU environments to train it.

**1. Push this project to your GitHub**
Make sure your latest codebase is pushed to your GitHub repository. Do **not** push the massive `data/` folder.

**2. Setup Kaggle**
Create a new Notebook in Kaggle and clone your repository directly into the Kaggle environment:
```python
!git clone https://github.com/Weptsugar/VaayuChakshu.git
%cd VaayuChakshu
!pip install -r requirements.txt
```

**3. Obtain the Dataset**
You will need the C2S-MS Floods dataset (or your own Sentinel-1/Sentinel-2 pairs). If you are using the original dataset, download the `data.zip` file (available via [Insert Link to your Kaggle Dataset or Google Drive here]).

**4. Upload and Prepare Dataset**
In Kaggle, click **Add Data -> Upload -> New Dataset**, and upload the `data.zip` file you downloaded. Kaggle will automatically extract it into `/kaggle/input/`. Copy it into the project and generate the manifest:
```python
# Copy extracted data from Kaggle Input to your project
!cp -r /kaggle/input/your-dataset-name/* data/raw/

# Generate the data_manifest.csv required for training
!python scripts/prepare_dataset.py
```

**4. Start Training!**
Run the training script. This will take roughly an hour on a T4 GPU.
```python
!python -m src.train --config config.yaml
```

---

## 🏢 Architectural Deep Dive

### The Multi-Task U-Net Generator
The generator is responsible for the core translation task. We employ a **U-Net architecture**, which is an encoder-decoder network enhanced with skip connections. A key innovation in our design is the **Multi-Task Output Head**. Instead of a single output, our U-Net generator's final layer branches into two separate heads:
1.  **RGB Head:** Generates the final optical image.
2.  **Segmentation Head:** Produces the raw binary flood water segmentation mask.

### The PatchGAN Discriminator
To enforce realism, we use a 70x70 **PatchGAN discriminator**. Instead of classifying the entire image as real or fake, the PatchGAN operates on `N x N` patches of the image. It outputs a feature map where each value represents the "realness" of a corresponding patch, encouraging the generator to produce realistic high-frequency details.

---

## 📊 The Physics-Aware Loss Function

The true power of VaayuChakshu comes from its multi-component loss function, meticulously engineered to guide the model using principles of remote sensing and SAR physics.

| Loss Component | Purpose & Implementation | Key Benefit |
| :--- | :--- | :--- |
| **`L_adversarial`** | Enforces realism by training the generator to fool the discriminator. | Provides smoother, more stable gradients. |
| **`L1_masked`** | Calculates L1 (Mean Absolute Error) between optical images, **only on clear pixels** defined by the cloud mask. | Learns from partially cloudy scenes without penalizing cloud generation. |
| **`Perceptual_masked`** | Compares high-level feature maps from a pre-trained VGG19 network. | Improves perceptual quality (texture and structure). |
| **`SegmentationLoss`** | **Directly supervises the water segmentation output.** | Makes the model an explicit flood mapper, not just an image translator. |
| **`SpecklePreservationLoss`** | Penalizes the generator for creating high-frequency details in the optical image where the corresponding SAR image is smooth (e.g., calm water). | Creates realistic water surfaces by leveraging physical properties of SAR. |
| **`WaterIndexConsistencyLoss`** | Calculates a proxy NDWI on the *generated optical image* and penalizes the model if it doesn't match the input SAR VH channel. | **Provides powerful cross-modal physics guidance.** Ensures the generated optical image is hydrologically consistent. |
