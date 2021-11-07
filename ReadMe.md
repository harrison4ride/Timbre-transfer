# Data prepare and preprocessing
Our paired data was generated from two dataset. Thus you need to down load it first
## The NSynth Dataset, “A large-scale and high-quality dataset of annotated musical notes.”
https://magenta.tensorflow.org/datasets/nsynth
## Classical Music MIDI Dataset
https://www.kaggle.com/soumikrakshit/classical-music-midi

Then, move the dataset to the path 'Timbre-transfer' and run the 'data_preprocessing.ipynb'. It will generated paired audio data, and paired spectrogram in CQT/STFT(time-frequency analysis methods).

# Train the model
Our model use a pre-trained model to achieve best performance. Thus you need to create path './models/keyboard_acoustic_2_string_acoustic_cqt' and './models/keyboard_acoustic_2_string_acoustic_stft'. Then, download the pre-trained model from https://drive.google.com/drive/folders/15qzaeFJ_vpRqPR_kevyOLIKobWQ6xhqC?usp=sharing and move it to the two path above. Then train the model with 'train_cqt.py' or 'train_stft.py'.

# Evaluation
The output of the model will be located in './data/outputs/', and the loss will be recorded in './model'. After training, you can run the 'evaluation.ipynb' in './data/outputs/' for evaluation. We apply cross-entropy, FID and audio distance for evaluation.