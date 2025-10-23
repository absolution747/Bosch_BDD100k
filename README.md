# 🚗 Bosch BDD100k Dataset Project

A comprehensive toolkit for working with the BDD100k dataset, featuring Docker containerization, automated dataset download, and in-depth data analysis capabilities.

---

## 🚀 Quick Start

### Building the Docker Environment
ā
Build and launch the Docker container with a single command:

```bash
bash build.sh
```

This command will:
- Build the Docker image with all necessary dependencies
- Launch a container with **port 8888** exposed
- Enable Jupyter Notebook access for interactive analysis

---

## 📥 Dataset Download

Download the BDD100k dataset using your unique access credentials:

```bash
bash download_data.sh <32-character-id>
```

**Parameters:**
- `<32-character-id>`: Your personal 32-character dataset access ID

Once downloaded, the dataset will be ready for:
- 📊 Data analysis
- 🧠 Model training
- 📈 Performance evaluation

---

## 📊 Data Analysis & Model Training

### Interactive Analytics Notebook

Explore comprehensive dataset insights through our dedicated analytics notebook:

**[📓 Open Analytics Visualizer Notebook](https://github.com/absolution747/Bosch_BDD100k/blob/main/data_analysis/Analytics_Visualizer.ipynb)**

### Accessing the Notebook Locally

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

2. Navigate to and open:
   ```
   data_analysis/Analytics_Visualizer.ipynb
   ```

3. Run the notebook cells to view:
   - Detailed dataset statistics
   - Visualization of data distributions
   - In-depth analysis and insights

### Generate subset of the Dataset
For proof of training we create a training subset of the dataset to have a more time realistic training. Here we create a mini dataset of 1000 images
```bash
   python3 data_analysis/generate_subset.py
   ```
apart from generating a subset of the data, the data is also in the YOLO format which is compatible for the training of the yolov8m model.

### Model Training
The model training script can be viewed here:

**[📓 Open Model Training Notebook](https://github.com/absolution747/Bosch_BDD100k/blob/main/training/train.ipynb)**

For detailed explainer on model choice, architecture, training parameters and augmentations please please view the following MarkDown.

**[📓 model_conclusion.md](https://github.com/absolution747/Bosch_BDD100k/blob/main/training/model_conclusion.md)**


---

## 📋 Features

- ✅ Dockerized environment for consistent setup
- ✅ Automated dataset download script
- ✅ Interactive Jupyter notebook for analysis
- ✅ Ready-to-use pipeline for training and evaluation
- ✅ Comprehensive data visualization tools

---

## 🛠️ Technology Stack

- **Docker** - Containerization
- **Jupyter Notebook** - Interactive analysis
- **BDD100k Dataset** - Large-scale driving dataset

---

## 📝 Notes

Make sure you have:
- Docker installed and running
- Valid BDD100k dataset access credentials
- Sufficient storage space for the dataset

---