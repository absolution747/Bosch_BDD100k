# 🚗 Bosch BDD100k Dataset Project

A comprehensive toolkit for working with the BDD100k dataset, featuring Docker containerization, automated dataset download, and in-depth data analysis capabilities.

---

## 🚀 Quick Start

### Building the Docker Environment

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

### Generate subset of Dataset
For proof of training we create a training/valdiation subset of the dataset to have a more time practical training. Here we create a mini dataset of 1000 images that cover each class category proportionally 
```bash
   python3 data/generate_subset.py
   ```
### Format conversion
we have opted for using the data to train a yolov8m model. For this reason we will need to convert the data format to coco standart format. We will need to execute the python script shown below
```bash
   python3 convert/bdd_to_coco.py
   ```
This script will generate the file ***/workspace/bdd100k_subset_1k/coco_annotations.json*** which will be used for training the yolov8m model

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