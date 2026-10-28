

\# SteelSight — Explainable Industrial Defect Classification



SteelSight is a deep-learning application that classifies steel surface images

into six defect categories using transfer learning with ResNet18. It combines

quantitative model evaluation with Grad-CAM visual explanations in a Streamlit

web interface.



\## Features



\- Six-class industrial steel defect classification

\- Transfer learning with ImageNet-pretrained ResNet18

\- GPU training using PyTorch and CUDA

\- Data augmentation and stratified train/validation splitting

\- Held-out test evaluation with accuracy, macro-F1, precision, and recall

\- Confusion matrix and per-class classification report

\- Grad-CAM heatmaps for qualitative model explainability

\- Interactive image-upload application built with Streamlit



\## Dataset



Dataset: NEU steel surface defect images (NEU-DET).



Dataset source:

https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database



The project uses six classes:



\- crazing

\- inclusion

\- patches

\- pitted\_surface

\- rolled-in\_scale

\- scratches



Expected local layout:



data/raw/NEU-DET/

├── train/images/

│   ├── crazing/

│   ├── inclusion/

│   ├── patches/

│   ├── pitted\_surface/

│   ├── rolled-in\_scale/

│   └── scratches/

└── validation/images/

&#x20;   ├── crazing/

&#x20;   ├── inclusion/

&#x20;   ├── patches/

&#x20;   ├── pitted\_surface/

&#x20;   ├── rolled-in\_scale/

&#x20;   └── scratches/



The data loader creates a reproducible stratified split from the provided

training images. The provided validation directory is reserved as the final

held-out evaluation set.



Raw dataset images are not included in this repository. Download the dataset

and place it in the directory structure shown above. Follow the dataset's

license and usage requirements.



\## Model architecture



\- Backbone: ResNet18

\- Initial weights: ImageNet pretrained

\- Input resolution: 224 × 224 RGB

\- Output classes: 6

\- Loss: Cross-entropy

\- Optimizer: AdamW

\- Model selection: best internal-validation macro-F1

\- Early stopping: validation macro-F1



The initial training strategy freezes early feature-extraction layers and

fine-tunes the final convolutional block and classification head.



\## Experimental results



Held-out evaluation on 360 images (60 images per class):



| Metric | Result |

|---|---:|

| Test accuracy | 99.44% |

| Test macro-F1 | 99.44% |

| Test cross-entropy loss | 0.0216 |

| Correct predictions | 358 / 360 |



Per-class results from the recorded evaluation:



| Class | Precision | Recall | F1-score |

|---|---:|---:|---:|

| Crazing | 1.000 | 1.000 | 1.000 |

| Inclusion | 1.000 | 0.967 | 0.983 |

| Patches | 1.000 | 1.000 | 1.000 |

| Pitted surface | 1.000 | 1.000 | 1.000 |

| Rolled-in scale | 1.000 | 1.000 | 1.000 |

| Scratches | 0.968 | 1.000 | 0.984 |



These results represent one held-out split of a relatively small dataset.

They do not establish performance on unseen factory environments, camera

systems, lighting conditions, or steel products.



\## Project structure



SteelSight/

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

├── data/

│   └── raw/                 # Download separately; not committed

├── src/

│   ├── \_\_init\_\_.py

│   ├── data.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   └── gradcam.py

├── models/

│   └── best\_model.pth

└── reports/

&#x20;   ├── training\_history.csv

&#x20;   ├── training\_curves.png

&#x20;   ├── test\_metrics.json

&#x20;   ├── test\_classification\_report.csv

&#x20;   ├── test\_confusion\_matrix.png

&#x20;   └── gradcam\_examples/



\## Setup



Create and activate a Python 3.11 environment:



&#x20;   conda create -n steelsight python=3.11 -y

&#x20;   conda activate steelsight

&#x20;   python -m pip install --upgrade pip



Install a compatible CUDA-enabled PyTorch build using the official selector:

https://pytorch.org/get-started/locally/



For the original development environment, the installed build was:



&#x20;   python -m pip install torch==2.14.1+cu132 torchvision==0.29.2+cu132 --index-url https://download.pytorch.org/whl/cu132



Install the other dependencies:



&#x20;   pip install streamlit scikit-learn matplotlib numpy Pillow



Verify GPU availability:



&#x20;   python -c "import torch; print(torch.cuda.is\_available()); print(torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')"



Ensure the dataset is in the expected local directory before training.



\## Train the model



&#x20;   python src\\train.py



This downloads the pretrained ResNet18 weights if needed and saves the best

checkpoint and training reports.



\## Evaluate the held-out dataset



&#x20;   python src\\evaluate.py



The test metrics and confusion matrix are written to `reports/`.



\## Generate Grad-CAM explanations



&#x20;   python src\\gradcam.py



Visualizations are saved under `reports/gradcam\_examples/`.



\## Launch the application



&#x20;   streamlit run app.py



Upload an image to view its predicted class, class scores, inference time,

and Grad-CAM explanation.



\## Limitations



\- The model was evaluated on a small public dataset, not a production

&#x20; inspection line.

\- Softmax outputs are not necessarily calibrated probabilities.

\- Grad-CAM provides qualitative explanations, not pixel-accurate defect masks.

\- Images from different cameras, factories, materials, and lighting

&#x20; conditions may produce different results.

\- This educational prototype should not be the sole basis for safety-critical

&#x20; quality-control decisions.



\## Future improvements



\- Evaluate robustness under realistic changes in lighting and image quality.

\- Compare ResNet18 against MobileNetV3 or EfficientNet.

\- Add confidence calibration and out-of-distribution detection.

\- Evaluate pixel-level anomaly localization using a suitable benchmark.

\- Package the application for reproducible deployment.



