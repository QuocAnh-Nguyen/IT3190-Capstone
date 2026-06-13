**Project Title:** Multi-modal Music Genre Classification
**Objective:** Build a comprehensive, end-to-end machine learning pipeline to predict a track's genres (Multi-Label Classification) by combining raw audio data processing with the semantic analysis of lyrics.

### Phase 1: Data Engineering & Preprocessing
* **Foundation Dataset:** Utilize **MusicOSet**, an open-source academic dataset containing multi-dimensional information on artists, tracks, and albums, structured as a relational database.
* **Data Integration:** Set up a local SQL database and engineer queries to merge the `Artists`, `Genres`, and `Lyrics` tables. We will bypass pre-computed acoustic features and map the database records directly to our locally stored raw audio files.
* **Missing Data Handling:** Implement a robust strategy for tracks with missing data (e.g., broken audio URLs, missing lyrics, or insufficient metadata). Depending on data availability, decide whether to drop records missing multiple modalities or impute/mask missing inputs so the model can still predict using a single available modality.
* **Class Imbalance (Multi-Label):** Address the dataset skew where certain genres (e.g., "Pop", "Rock") dominate. Utilize multi-label specific sampling techniques (like MLSMOTE) or adjust class weights in the loss function to ensure the model penalizes majority-class bias.

### Phase 2: Advanced Feature Engineering
We will build a dual-track feature extraction pipeline, comparing traditional ML feature extraction against modern Deep Learning embeddings:

* **Audio Processing (Two-Pronged Approach):**
  * *Traditional Methods:* Use libraries like `librosa` to extract mathematical audio features (e.g., MFCCs, Mel-Spectrograms, Chroma features, Spectral Contrast).
  * *Pre-trained Deep Learning Models:* Pass the raw audio through pre-trained audio encoders (e.g., VGGish, AST - Audio Spectrogram Transformer, or Wav2Vec) to generate dense audio embeddings.
* **NLP on Lyrics (Two-Pronged Approach):**
  * *Traditional Methods:* Transform lyrics using classic text processing (TF-IDF, Bag of Words) combined with Lexical Richness metrics and Sentiment Analysis scores.
  * *Pre-trained Deep Learning Models:* Utilize transformer-based text encoders (e.g., BERT, RoBERTa) to extract rich semantic embeddings from the lyrics.
* **Feature Selection & Dimensionality Reduction:** For the traditional feature routes, apply Principal Component Analysis (PCA) or Tree-based Feature Importance to prevent the "Curse of Dimensionality" before fusion.

### Phase 3: Model Building & Evaluation
* **Hybrid Architecture (Late Fusion):** 
  * *Stream 1:* Process the Audio vectors (Traditional or Pre-trained Embeddings).
  * *Stream 2:* Process the Text vectors (Traditional or Pre-trained Embeddings).
  * *Fusion:* Concatenate both streams into a unified representation, feeding into a final classification head.
* **Algorithms (Multi-Label Classification):** Since a single track can belong to multiple genres simultaneously, shift from multi-class to multi-label architectures. 
  * *Traditional/Ensemble:* Multi-label Random Forest, Classifier Chains, or XGBoost configured for multi-label output.
  * *Neural Network:* A Multi-Layer Perceptron (MLP) or custom Neural Network fusion layer utilizing a **Sigmoid** activation function on the output layer with a **Binary Cross-Entropy (BCE)** loss function.
* **Evaluation Metrics:** Validate the model using $K$-Fold Cross-Validation adapted for multi-label datasets. Track metrics specific to multi-label performance: **Macro/Micro F1-Score**, **Hamming Loss**, **Exact Match Ratio (Subset Accuracy)**, and generate a Multi-label Confusion Matrix.