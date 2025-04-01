# Medical Super Image Resolution using GAN

## 📌 Overview
Super-Resolution GAN (SRGAN) is a deep learning model designed to upscale low-resolution images to high resolution using Generative Adversarial Networks (GANs). It leverages residual and inception blocks for improved feature extraction and high-quality image reconstruction.

## 🚀 Features
- **GAN-based Super-Resolution:** Uses a generator and discriminator network for image enhancement.
- **Advanced Model Architecture:** Integrates residual and inception blocks for superior feature extraction.
- **Optimized Training:** Fine-tuned using PSNR and SSIM metrics for high-quality output.
- **End-to-End Processing:** Supports image loading, preprocessing, training, and upscaling.

## 🛠️ Tech Stack
- **Frameworks & Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Languages:** Python

## 📂 Dataset
- The project uses images from the [Urban100 dataset](https://www.kaggle.com/datasets/harshraone/urban100), containing high-resolution urban scene images.
- **Input:** Low-resolution images (64x64)
- **Output:** High-resolution images (256x256)

## 🏗️ Model Architecture
- **Generator:**
  - Uses residual and inception blocks to enhance image features.
  - Upsamples images using transposed convolutions.
- **Discriminator:**
  - Classifies images as real (high-res) or fake (generated).
  - Uses convolutional layers with LeakyReLU activation.

## 🔧 Installation
```sh
pip install tensorflow keras imageio opencv-python numpy matplotlib
```

## 📜 Usage
### 1️⃣ Load and Preprocess Data
```python
high_res_images = load_images(high_res_input_dir)
low_res_images = load_images(low_res_input_dir, size=(64, 64))
high_res_images = preprocess(high_res_images)
low_res_images = preprocess(low_res_images)
```

### 2️⃣ Train the Model
```python
train_srgan(epochs=70, batch_size=16)
```

### 3️⃣ Upscale Images
```python
test_images = low_res_images[:5]
upscaled_images = upscale_images(test_images)
```

## 📊 Evaluation
- Performance is measured using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)** to compare generated images with ground truth high-resolution images.

## 🏆 Results
- The model significantly enhances image quality by reconstructing finer details and sharper textures.

## 📌 Future Improvements
- Fine-tune the model with larger datasets.
- Implement additional loss functions like perceptual loss for better visual quality.
- Optimize inference speed for real-time applications.

## 🤝 Contributing
Feel free to fork the repository, submit issues, or contribute improvements!

## 📜 License
This project is open-source and available under the **MIT License**.

