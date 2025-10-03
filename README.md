# Fine-Tuned

An Intuitive Desktop GUI for Fine-Tuning Language Models.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Framework](https://img.shields.io/badge/Framework-PySide6-cyan.svg)
![Backend](https://img.shields.io/badge/Backend-Hugging_Face-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/dovvnloading/Fine-Tuned?style=social)](https://github.com/dovvnloading/Fine-Tuned/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/dovvnloading/Fine-Tuned?style=social)](https://github.com/dovvnloading/Fine-Tuned/network/members)

---

**Fine-Tuned** is a desktop application designed to bridge the gap between powerful language model fine-tuning and a user-friendly graphical interface. Built with PySide6 and powered by the Hugging Face ecosystem (`transformers`, `datasets`), this tool allows researchers, developers, and enthusiasts to train models on custom data without writing complex scripts.

The application provides real-time feedback, visualizes training progress, and offers a suite of configuration options, making the experimental process of fine-tuning more accessible and efficient.

<img width="1202" height="1032" alt="Screenshot 2025-10-03 162053" src="https://github.com/user-attachments/assets/fabeefa5-c57d-4913-a395-82036a0dfc73" />
<img width="1202" height="1032" alt="Screenshot 2025-10-03 163250" src="https://github.com/user-attachments/assets/07ec593e-8bf6-4469-aa5d-727406b6aee4" />


## ✨ Features

-   **✨ Intuitive GUI**: A clean, modern, and responsive interface built with PySide6, featuring a logical workflow from configuration to inference.
-   **📈 Real-time Training Visualization**: A live-updating Matplotlib chart plots training and validation loss, giving you immediate insight into model performance.
-   **🧠 Smart Stopping**: Integrated Early Stopping callbacks to prevent overfitting and save time by automatically halting training when the model's validation performance plateaus.
-   **⚙️ Granular Hyperparameter Control**: Easily adjust key parameters like epochs, batch size, learning rate, gradient accumulation, and scheduler type through accessible UI controls.
-   **🚀 Built-in Inference Panel**: Test your base model or the newly fine-tuned model directly within the app. Enter a prompt and get an immediate generation.
-   **🤖 Hugging Face Integration**: Seamlessly loads pre-trained GPT-2 family models and tokenizers from the Hugging Face Hub.
-   **🌐 Responsive, Non-Blocking UI**: Training and model loading are performed on separate threads, ensuring the application remains responsive at all times.

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

-   Python 3.8+
-   Git
-   **NVIDIA GPU with CUDA**: While the application can run on a CPU, training language models is computationally intensive. A CUDA-enabled GPU is **strongly recommended** for a feasible training experience.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dovvnloading/Fine-Tuned.git
    cd Fine-Tuned
    ```

2.  **Create and activate a virtual environment:**
    -   On **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   On **macOS / Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    This project requires PyTorch with CUDA support. It's crucial to install this first, following the official instructions.

    -   **Step A: Install PyTorch**
        Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and select the appropriate settings for your system (e.g., Stable, Windows/Linux, Pip, Python, your CUDA version). Then run the generated command. For example:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    -   **Step B: Install other project requirements**
        A `requirements.txt` file is provided for your convenience.
        ```bash
        pip install -r requirements.txt
        ```

## 🎮 How to Use

### 1. Prepare Your Dataset

The application expects a JSON or JSON Lines (`.jsonl`) file where each line is a JSON object containing `"question"` and `"answer"` keys.

**Example `dataset.json`:**
```json
[
  {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
  {"question": "Explain the theory of relativity in simple terms.", "answer": "Einstein's theory of relativity has two parts. Special relativity says that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is the same for everyone. General relativity explains gravity as a curvature of spacetime caused by mass and energy."},
  {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee wrote the classic novel 'To Kill a Mockingbird'."}
]
```

### 2. Launch the Application

Run the main Python script from the project root:
```bash
python main.py
```

### 3. Configure the Training Run

-   **Section 1: Configuration**:
    -   Use the **Browse** buttons to select your dataset file and specify an output directory for the trained model.
    -   Choose a base model from the dropdown (e.g., `distilgpt2`). You can load it for testing before training.

-   **Section 2: Training Parameters**:
    -   Enable and expand this section to fine-tune hyperparameters. Defaults are provided for a quick start.

-   **Section 3: Smart Stopping**:
    -   Keep this enabled to use validation data for early stopping. Configure the validation set size, evaluation frequency, and patience.

### 4. Start Training

-   Click the **"Start Training"** button.
-   The UI will lock, and progress will be displayed in the **Training Progress** log on the right.
-   The loss chart will update in real-time.

### 5. Test Your Model

-   Once training is complete, the best model is automatically loaded into the **"Test Your Model"** panel.
-   Enter a question in the input field and click **"Ask"** to see your model's response.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or encounter a bug, please feel free to:

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

You can also open an [issue](https://github.com/dovvnloading/Fine-Tuned/issues) with the "bug" or "enhancement" tag.

## ⚖️ License

This project is distributed under the Apache License 2.0. See the `LICENSE` file for more information.

---
