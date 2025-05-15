# Flickr8k-Image-Captioning-Using-CNNs-and-LSTMs

### What is Image Captioning?

- Image Captioning is the process of generating a **textual description** of an image.
- It lies at the intersection of **Computer Vision** (for understanding image content) and **Natural Language Processing** (for generating descriptive text).
- A typical image captioning system uses an **encoder-decoder architecture**:
  - **Encoder (CNN)** converts the input image into a feature representation.
  - **Decoder (LSTM)** takes these features and generates a sentence, word-by-word.

### Why CNN + LSTM?

- We combine two deep learning models:
  - **CNN (e.g., ResNet, VGG)** for extracting deep features from an image.
  - **LSTM (Long Short-Term Memory)** for generating a natural language sequence.
- The image features are first encoded using CNN, then passed to an LSTM along with word embeddings.
- The LSTM predicts the next word in the caption at each time step, conditioned on the previous words and the image context.

> 🔍 This fusion of vision and language makes it possible to generate coherent, accurate, and context-aware descriptions of images.

### Architecture Overview

![Architecture](https://miro.medium.com/max/1400/1*6BFOIdSHlk24Z3DFEakvnQ.png)

> This image represents a simplified flow of how the image passes through a CNN, then to an LSTM along with word sequences, eventually producing a caption.


Here is the revised Markdown report for **Sections 2 to 7**, now with code blocks inserted directly within each relevant section, under the appropriate paragraph — no separate file, all inline and structured:

---

## Section 2: Preprocessing and Tokenization

Before feeding the captions into a neural network, the text must be cleaned and structured. This process is known as **text preprocessing**, and it's a critical step in any NLP pipeline. In this project, all captions were converted to lowercase to ensure case consistency, and special characters were removed to reduce noise. Redundant white spaces and standalone characters were also filtered out.

<img src='https://lena-voita.github.io/resources/lectures/word_emb/lookup_table.gif'>

To help the model identify the beginning and end of each caption, two special tokens — `startseq` and `endseq` — were appended to every caption. These tokens serve as markers that guide the sequence model during training and inference.

Once cleaned, the captions are tokenized — that is, each word is mapped to a unique integer. This creates a vocabulary that the model can work with. We also calculate the maximum sequence length for use in later padding.

```python
data = text_preprocessing(data)
captions = data['caption'].tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)
```

Next, the dataset is split into training and validation sets using an 85:15 split on unique image filenames.

```python
images = data['image'].unique().tolist()
nimages = len(images)
split_index = round(0.85*nimages)
train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)
```

---

## Section 3: Image Feature Extraction with CNN

Raw images are high-dimensional and not directly suitable as input to an LSTM. Instead, we extract **feature vectors** using a pre-trained CNN — here, **DenseNet201**. DenseNet’s final classification layers are removed, and the penultimate layer is used to provide a rich, fixed-length descriptor of each image.

![DenseNet Architecture](https://imgur.com/wWHWbQt.jpg)

```python
model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

img_size = 224
features = {}
for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)
    features[image] = feature
```

This process transforms each image into a vector of size 1920, which acts as the visual input to the decoder.

---

## Section 4: Data Generation (Batch Feeding)

Instead of loading the entire dataset at once, we use a **custom Keras data generator**. This generator feeds image features and caption sequences to the model in batches, optimizing memory usage.

Each image-caption pair is expanded into multiple training examples where the model sees partial captions and tries to predict the next word.

```python
class CustomDataGenerator(Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer, 
                 vocab_size, max_length, features,shuffle=True):
    
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __getitem__(self,index):
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        X1, X2, y = self.__get_data(batch)        
        return (X1, X2), y
    
    def __get_data(self,batch):
        
        X1, X2, y = list(), list(), list()
        
        images = batch[self.X_col].tolist()
           
        for image in images:
            feature = self.features[image][0]
            
            captions = batch.loc[batch[self.X_col]==image, self.y_col].tolist()
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1,len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                
        return X1, X2, y
```

This generator provides:

* `X1`: image features
* `X2`: caption input sequence
* `y`: target word (as one-hot)

---

## Section 5: Model Architecture – CNN + LSTM Integration
Model Structure
<img src='https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png'>

We now define a model that combines the CNN-extracted image features with embedded caption sequences using an LSTM decoder. This is a simplified **encoder–decoder architecture**:

1. The image feature vector is passed through a dense layer.
2. The input caption is embedded and fed into an LSTM.
3. The outputs are combined and passed through additional dense layers to produce predictions.

```python
input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped, sentence_features], axis=1)

sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1, input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
```

This architecture ensures both visual and textual contexts are used to predict the next word in the sequence.

## **Model Modification**
- A slight change has been made in the original model architecture to push the performance. The image feature embeddings are added to the output of the LSTMs and then passed on to the fully connected layers
- This slightly improves the performance of the model orignally proposed back in 2014: __Show and Tell: A Neural Image Caption Generator__ (https://arxiv.org/pdf/1411.4555.pdf)

![image](https://github.com/user-attachments/assets/ef417412-3e69-40fe-a09a-8fc214cdc74a)

---

## Section 6: Training Strategy and Callbacks

To ensure stable training, we use **callbacks** like early stopping and learning rate scheduling. These help prevent overfitting and adjust the learning rate if training stalls.

```python
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)
```

We then begin model training using the custom data generators defined earlier.

```python
history = caption_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystopping, learning_rate_reduction]
)
```

![Training Loss Curve](https://github.com/user-attachments/assets/b94135f8-3a43-45f1-8990-e60bd4896403)

  
---
## Section 7: Inference and Caption Generation

At inference time, we generate captions by:

1. Extracting the image feature.
2. Starting the caption with `startseq`.
3. Iteratively predicting the next word and appending it to the sequence.
4. Stopping at `endseq` or when the caption reaches the maximum allowed length.

```python
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)

        if word is None or word == 'endseq':
            break
        in_text += " " + word

    return in_text
```

This method is known as **greedy decoding**. For more advanced results, **beam search** can be used to explore multiple possible sequences and select the best one.
---

## Section 8: Caption Prediction and Evaluation

After training the model, the next step is to evaluate its performance by generating captions for unseen images from the validation set. To do this, 15 random samples were selected from the test set. Each image was passed through the model, and a caption was predicted using the greedy decoding function defined earlier.



This section helps assess how well the model has learned to associate visual features with coherent language descriptions.

```python
samples = test.sample(15)
samples.reset_index(drop=True, inplace=True)
```

For each image:

* The image is loaded and preprocessed to the appropriate shape and normalization.
* The model generates a caption using the `predict_caption()` function.
* The generated caption is stored for visualization.

```python
for index, record in samples.iterrows():
    img = load_img(os.path.join(image_path, record['image']), target_size=(224, 224))
    img = img_to_array(img) / 255.0
    caption = predict_caption(caption_model, record['image'], tokenizer, max_length, features)
    samples.loc[index, 'caption'] = caption
```

### Visualization of Predictions

Once predictions are made, the images and their generated captions are displayed in a grid format. This visualization offers a direct way to observe how meaningful, fluent, and relevant the model's outputs are.

![image](https://github.com/user-attachments/assets/d4b42541-a7f0-414f-b5b7-e44688b367ed)

---

## Section 9: Results, Conclusion & Future Work

### Observations

* Some captions are accurate and meaningful, but certain patterns of **repetition and bias** were noted.
* For example, phrases like “dog running through the water” appeared repeatedly, even for images with no dogs.
* Similarly, “man in a blue shirt” was overused, highlighting that the model may be generalizing visual cues improperly.

This behavior indicates that while the model captures a basic visual-language relationship, it lacks **finer attention to object details and color** differentiation. This is expected when training on relatively small datasets like Flickr8k.

### Conclusion

> *“This may not be the best performing model, but the objective of this project was to demonstrate the pipeline and design of an image captioning system. The results show that even a basic CNN-LSTM architecture can learn meaningful visual-to-language mappings.”*

By understanding this pipeline — from preprocessing, tokenization, feature extraction, and caption generation to evaluation — a strong foundation is laid for future improvements and experimentation.

---

### Future Improvements

The current model can be significantly enhanced using more advanced techniques:

1. **Attention Mechanism**
   By allowing the model to focus on different parts of the image at different times, attention mechanisms can reduce redundancy and improve contextual awareness. They also make the model interpretable — we can visualize which region of the image contributes to which word.

2. **BLEU and Other Evaluation Metrics**
   In the next phase, we can use evaluation metrics like BLEU, METEOR, and CIDEr to quantify caption quality. These metrics compare generated captions with reference captions and provide objective scores for evaluation.

3. **Larger Datasets**
   Training the same model on richer datasets like MS-COCO or Flickr30k would give it more diverse training examples and improve generalization.

4. **Transformers for Vision-Language Tasks**
   Emerging models like **BLIP**, **OSCAR**, and **ClipCap** use vision-language transformers instead of LSTMs. These models achieve state-of-the-art results and may be integrated into future iterations for more accurate and creative captioning.

---

## Section 10: Evaluation Metrics – BLEU, METEOR, CIDEr, ROUGE

Evaluating image captioning models is challenging because **multiple correct captions** may exist for a single image. Human-written descriptions vary in phrasing, word order, and detail. Thus, traditional classification accuracy is insufficient. Instead, specialized **text similarity metrics** are used to compare the model-generated caption to one or more reference (ground truth) captions.

Below are the most widely used metrics in image captioning tasks:

---

### **BLEU (Bilingual Evaluation Understudy)**

* **What it measures**: N-gram precision (overlap of unigrams, bigrams, trigrams, etc.) between the candidate and reference captions.
* **How it works**: Counts how many n-grams in the predicted caption appear in the references.
* **Key feature**: Includes a brevity penalty to penalize short, overly simple captions.

```python
from nltk.translate.bleu_score import sentence_bleu

score = sentence_bleu([reference_tokens], predicted_tokens, weights=(0.5, 0.5))
```

* **BLEU-1** uses unigrams (individual words), BLEU-4 uses up to 4-grams.
* **Typical Flickr8k scores**: BLEU-1 ≈ 0.6, BLEU-4 ≈ 0.2–0.3 for decent models.

> Best used for: Basic overlap checks. High BLEU means high phrase match.

---

### **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**

* **What it measures**: F1 score of matched words, including synonyms, stems, and paraphrases.
* **How it works**: Aligns the predicted caption to reference captions using flexible word matching and calculates harmonic mean of precision and recall.
* **Advantages**: More semantically aware than BLEU; handles word order, synonyms, and stems.

```python
from nltk.translate.meteor_score import meteor_score

score = meteor_score([ref_caption], predicted_caption)
```

* **Typical range**: 0.2–0.4 for Flickr8k.

> Best used for: Captions where semantics matter more than exact word matches.

---

### **CIDEr (Consensus-based Image Description Evaluation)**

* **What it measures**: Consensus with human-written captions using TF-IDF weighted n-grams.
* **How it works**: Gives higher importance to n-grams that appear frequently in references but rarely across the dataset.
* **Why it’s great**: Designed specifically for image captioning tasks.

CIDEr is typically computed using `pycocoevalcap` (COCO Caption Evaluation toolkit).

> Best used for: Large datasets (e.g., MS-COCO), rewards distinctive yet accurate captions.

---

### **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**

* **What it measures**: Longest common subsequence between predicted and reference captions.
* **Focus**: Recall — how much of the reference caption is covered by the generated one.
* **Good for**: Measuring fluency and recall-oriented metrics.

```python
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(predicted_caption, reference_caption)
```

> Best used for: Evaluating how complete a generated caption is.

---

### Metric Comparison Summary

| Metric  | Focus     | Strength                       | Weakness                    |
| ------- | --------- | ------------------------------ | --------------------------- |
| BLEU    | Precision | Simple, widely used            | Ignores meaning/synonyms    |
| METEOR  | F1 Score  | Handles synonyms, order, stems | Slightly slower to compute  |
| ROUGE-L | Recall    | Captures structure/order       | Not designed for short text |
| CIDEr   | Consensus | Best for image captioning      | Needs multiple references   |

---

### Applying Metrics in Practice

To apply these metrics, ensure you have:

* Tokenized reference captions (list of 5 ground-truth captions per image).
* Tokenized predicted captions (generated by your model).
* Metric functions (use `nltk`, `pycocoevalcap`, or `rouge` libraries).


---
Here’s the final section of your report, written in clear, professional Markdown. This wraps up the project, reflects on the learning and results, and positions it well for portfolio or academic presentation.

---

## Section 11: Final Project Summary

### Project Objective

This project demonstrates the complete pipeline for building an **image captioning model** using a **CNN-LSTM architecture** on the **Flickr8k** dataset. The goal was not only to generate captions but also to deeply understand each component involved — from data preprocessing and feature extraction to sequence modeling and evaluation.

---

###  Key Components Built

* **Text Cleaning and Tokenization**: Cleaned and structured over 40,000 captions into machine-readable sequences.
* **Image Feature Extraction**: Leveraged a pre-trained DenseNet201 model to convert images into dense, high-dimensional feature vectors.
* **Sequence Generator Model**: Designed a custom neural network that combines image features with LSTM-based sequence generation.
* **Custom Training Pipeline**: Used Keras generators and callbacks to efficiently train the model on limited memory.
* **Caption Generation Inference**: Implemented greedy decoding for caption prediction, supported by utility functions.
* **Result Visualization**: Generated and displayed captions for 15 unseen images, revealing strengths and limitations of the model.

---

### Outcomes and Insights

* The model successfully learned to associate visual patterns with coherent phrases.
* Some common errors included **repetition**, **bias toward certain objects (e.g., dogs, men)**, and **hallucinations** (describing things not in the image).
* The architecture shows promise, but performance could be significantly improved with more data and advanced techniques like attention.

---

### What Was Learned

* How to convert unstructured image and text data into a supervised learning format.
* How to extract and reuse visual features using CNNs.
* How to use LSTMs to model sequential text generation tasks.
* The importance of evaluation metrics in image captioning, especially BLEU and METEOR.
* Limitations of greedy decoding and benefits of beam search or attention-based alternatives.

---

### Future Scope and Recommendations

To take this project further:

1. **Use MS-COCO or Flickr30k**: These datasets offer more variety and volume.
2. **Implement Beam Search**: Greedy decoding can be limiting. Beam search allows exploration of better captions.
3. **Apply Attention Mechanism**: Improves context-awareness by focusing on relevant parts of the image for each word.
4. **Upgrade to Transformers**: Use recent models like BLIP, ClipCap, or OFA for state-of-the-art performance.
5. **Incorporate Evaluation Metrics**: Automate BLEU, METEOR, CIDEr evaluation to quantify progress during training.








