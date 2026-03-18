# DeceptionSense: Real-Time Audio Lie Detection 🎙️🕵️‍♂️

## 📌 Project Overview
**DeceptionSense** is an advanced artificial intelligence system designed to detect deception through real-time speech analysis. Unlike traditional polygraphs that rely on physiological stress sensors, this project utilizes a **Hybrid Deep Learning** architecture to classify spoken statements as truthful or deceptive purely based on acoustic and semantic patterns. By fusing handcrafted audio features with cutting-edge transformer embeddings, the system functions as a live, vocal forensics tool.

## 🚀 Key Features
* **Real-Time Inference:** Features an integrated pipeline utilizing `sounddevice` to capture live microphone input, extract features, and return a binary truth/lie prediction in seconds.
* **Hybrid Feature Extraction:** Combines the mathematical precision of **MFCCs** (Mel-frequency cepstral coefficients) with the deep contextual embeddings of Facebook's **Wav2Vec2** transformer model.
* **Multi-Domain Datasets:** Trained and evaluated on highly complex, real-world data, including the **Courtroom Testimonies** dataset and the **Dolos** behavioral speech dataset.
* **Automated Video Parsing:** Includes built-in `moviepy` scripts to automatically strip and resample audio tracks from raw interrogation or testimony video files.

---

## 🏗️ Architecture Details

### The Audio Processing Pipeline (`Feature_Extractor`)
The system maps raw audio waveforms into a structured numerical array suitable for classification.
* **Acoustic Profiling:** Uses `librosa` to extract 13 MFCCs, capturing physical speech characteristics like vocal tension, pitch variations, and micro-pauses.
* **Semantic Embeddings:** Passes the raw audio through the pretrained `Wav2Vec2Model` to extract the `last_hidden_state`, capturing the deep, linguistic context of the spoken words.
* **Feature Fusion:** Concatenates the averaged MFCCs and the Wav2Vec2 tensors into a single, high-dimensional feature vector.

### The Predictive Classifier (`MLP_Net`)
The concatenated vector is fed into a **Multi-Layer Perceptron (MLP)** for binary classification (Truth vs. Deception).
* **Network Structure:** Employs a dense architecture with hidden layers of `(128, 64)` to process the complex audio embeddings.
* **Optimization:** Runs up to 500 maximum iterations to ensure convergence on the highly nuanced differences between truthful and deceptive speech patterns.

---

## ⚖️ Training & Inference Logic
In this framework, the model ($M$) is trained to recognize the acoustic anomalies associated with deception:

1.  **Signal Normalization:** Audio is standardized to a `16000 Hz` sample rate, ensuring consistency whether the data comes from a downloaded YouTube courtroom video or live microphone input.
2.  **Model Optimization:** The network updates its weights using **Binary Cross-Entropy (Log-Loss)**. The model learns to output a probability ($\hat{y}$) indicating how likely a given audio sample is deceptive, compared to the actual truth label ($y$).

$$Loss = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

---

## 📈 Evaluation Metrics & Limitations
This project highlights the challenges and realities of behavioral machine learning:

* **Baseline Accuracy:** Achieved a moderate accuracy of **68%** on the highly complex courtroom dataset, establishing a strong proof-of-concept for acoustic deception detection.
* **Behavioral Nuance:** The evaluation exposed the inherent difficulty of the domain, noting a tendency for the model to misclassify truthful statements due to overlapping acoustic features (e.g., general nervousness vs. active deception). 
* **Future Roadmap:** Lays the groundwork for integrating BiLSTMs or attention-based transformers to better capture the temporal progression of a lie over a longer sentence.

---

## 👥 Contributors
* **Anik Basu** 
