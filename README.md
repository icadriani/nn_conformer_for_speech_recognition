**<h1>Semi-Supervised Learning for Automatic Speech Recognition</h1>**

This project aims to recognize speech through the use of conformers. The model is trained using supervised learning. The model is fine-tuned by applying Noisy Student Training (NST) with SpecAugment. The SpecAugment is applied to the data before training. During this phase the supervised dataset is mixed with an unsupervised dataset where the audio transcripts are generated from the model itself before training. This process is applied a number of times.

**<h2>Installation</h2>**

This project uses Python 3.9. Run the following command on the terminal to install the required libraries

`pip install -r requirements.txt`