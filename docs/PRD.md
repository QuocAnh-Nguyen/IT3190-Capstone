## REVISED PROJECT PROPOSAL

**Project Title:** Multi-modal Music Genre Classification
**Objective:** Build a comprehensive, end-to-end machine learning pipeline to predict a track's genre by exploiting complex relational data structures, combining numerical acoustic features with the semantic analysis of lyrics.

### Phase 1: Data Engineering & Preprocessing

* **Foundation Dataset:** Utilize **MusicOSet**, an open-source academic dataset containing multi-dimensional information on artists, tracks, and albums, structured as a relational database.
* **Data Integration:** Set up a local SQL database and engineer complex `JOIN` queries to merge the `AcousticFeatures`, `Artists`, `Genres`, and `Lyrics` tables into a unified Feature Matrix.
* **Advanced Imputation:** Instead of simply dropping missing values caused by incomplete cross-platform mapping (e.g., Spotify to Genius), implement a secondary machine learning model (Regression Imputation) to predict and fill in missing acoustic features.
* **Class Imbalance Handling:** Address the severe dataset skew toward "Pop" and "Rock" genres by applying advanced data sampling techniques, such as SMOTE (Synthetic Minority Over-sampling Technique) or targeted undersampling, to ensure an unbiased model.

### Phase 2: Advanced Feature Engineering

Rather than relying solely on raw data, the team will actively transform and synthesize new dimensions to provide the model with deeper insights:

* **Graph/Network Features:** Utilize Graph Theory (via `NetworkX`) to map collaborative tracks. By calculating network metrics like Degree Centrality, the model will capture the "homophily" (the tendency of similar artists to collaborate) within specific genres.
* **Advanced NLP on Lyrics:** Transform the textual `Lyrics` data into mathematical metrics by measuring Lexical Richness (unique word ratio) and performing Sentiment Analysis to capture the emotional tone of the track.
* **Interaction Features:** Apply `PolynomialFeatures` to cross-multiply acoustic metrics (e.g., `energy` $\times$ `danceability`). This helps linear models capture complex, non-linear boundaries between genres like EDM and Acoustic Pop.
* **Unsupervised Learning (Clustering):** Run a $K$-Means clustering algorithm on the acoustic data to group tracks into objective "Acoustic Clusters," completely independent of human-assigned genres. Use the resulting Cluster IDs as a brand new categorical feature.
* **Feature Selection:** Prevent the "Curse of Dimensionality" by utilizing algorithms like Random Forest Feature Importance or Principal Component Analysis (PCA) to retain only the most predictive variables.

### Phase 3: Model Building & Evaluation

* **Hybrid Architecture:** * *Stream 1:* Process the normalized numerical matrix (Acoustic + Graph + Interaction features).
* *Stream 2:* Process the text vectors (NLP features extracted from Lyrics).
* *Fusion:* Concatenate both streams to output the final genre prediction.


* **Algorithms:** Establish a robust baseline using Logistic Regression, then scale up to powerful ensemble models (Random Forest, XGBoost) or a Multi-Layer Perceptron (MLP) neural network.
* **Evaluation Metrics:** Validate the model using $K$-Fold Cross-Validation. Track Accuracy, Precision, Recall, and F1-Score (Macro/Micro for multi-class). Generate a Confusion Matrix to analyze misclassifications between acoustically similar genres.

### Phase 4: System Implementation & Deployment

* **Interface:** Develop a lightweight, interactive web application using **Streamlit** or **Gradio**.
* **Functionality:** Allow the user (instructor) to input hypothetical acoustic parameters or paste a snippet of song lyrics. The app will run the inputs through the entire preprocessing pipeline and display the predicted genre in real-time.
* **Explainability (Bonus):** Integrate **SHAP** (SHapley Additive exPlanations) values into the dashboard to visually explain *why* the model made its specific prediction, breaking down the exact impact of each feature.