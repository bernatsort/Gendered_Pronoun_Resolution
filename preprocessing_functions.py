# Text Cleaning Functions

def select_imp_features(df):
    imp_features = ["Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "B", "B-offset"]
    target_col = ["A-coref", "B-coref"]
    df = df[imp_features + target_col]
    return df 

def lower_case(df):
    df.loc[:, "text_clean"] = df["Text"].apply(lambda x: x.lower())
    df.loc[:, "Pronoun"] = df["Pronoun"].apply(lambda x: x.lower())
    df.loc[:, "A"] = df["A"].apply(lambda x: x.lower())
    df.loc[:, "B"] = df["B"].apply(lambda x: x.lower())

def expand_contractions(df):
    df.loc[:, "text_clean"] = df["text_clean"].apply(lambda x: contractions.fix(x))

def remove_non_ascii_characters(df, col='text_clean'):
    df.loc[:, col] = df[col].apply(lambda text: re.sub(r'[^\x00-\x7f]', r'', text)) # get rid of non-characters and whitespace
    return df

def remove_punctuations(df, col='text_clean'):
    """
     - str.maketrans('', '', string.punctuation) crea un traductor utilizando maketrans 
       que mapea los caracteres de puntuación a None, es decir, los elimina.
     - string.punctuation es una cadena predefinida en el módulo string que contiene todos 
       los caracteres de puntuación.
     - text.translate(translator) aplica el traductor al texto, reemplazando las puntuaciones 
       con caracteres vacíos, lo que efectivamente las elimina.
    """
    df.loc[:, col] = df[col].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
    # return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
    return df

# Orquestación secuencial de la limpieza de texto
def text_cleaning(df):
    # Crear una copia del DataFrame:
        # todas las modificaciones se realicen en una copia independiente 
        # y no afectan al DataFrame original.
    df = df.copy()  
    # text cleaning functions
    df = select_imp_features(df)
    lower_case(df)
    expand_contractions(df)
    df = remove_non_ascii_characters(df)
    df = remove_punctuations(df)
    return df

# Text Preprocessing Functions 

# Tokenization
def tokenize_text(df):
    df.loc[:, 'tokenized'] = df['text_clean'].apply(word_tokenize)

# Part of Speech Tagging
def part_of_speech_tagging(df):
    df.loc[:, 'pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('POS'):
        return wordnet.NOUN
    elif tag.startswith('RB'):
        return wordnet.ADV
    elif tag.startswith('CD'):
        return wordnet.NOUN   
    else:
        return wordnet.NOUN  # Por defecto, asumimos sustantivos si la etiqueta POS no coincide con las anteriores

def apply_wordnet_pos(df):
    df.loc[:, 'wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    
# Lemmatization with PoS
def lemmatize_word_pos(text):
    """
    Lemmatize the tokenized words (with PoS)
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word, tag) for word, tag in text]
    return lemmatized_text

def apply_lemmatize_word_pos(df):
    df.loc[:, 'lemmatize_word_pos'] = df['wordnet_pos'].apply(lambda x: lemmatize_word_pos(x))

# Orquestación secuencial del preprocesamiento del texto
def text_preprocessing(df):
    df = df.copy()

    tokenize_text(df)
    part_of_speech_tagging(df)
    apply_wordnet_pos(df)
    apply_lemmatize_word_pos(df)

    return df


# Text Features Extraction

# Word2Vec Embedding
def train_word2vec_model(sentences):
    # Set up the parameters of the model
    word2vec_model = Word2Vec(vector_size=300, 
                              window=2, 
                              min_count=20, 
                              sample=6e-5, 
                              workers=cores-1, 
                              sg=1)
                              
    # Building the Vocabulary Table
    word2vec_model.build_vocab(sentences)
    
    # Training the model
    word2vec_model.train(sentences, 
                         total_examples=word2vec_model.corpus_count,  # 2000: len(sentences)
                         epochs=30, 
                         report_delay=1)
    
    return word2vec_model

# Text Vectorization
def get_sentence_vector(words, word2vec_model):
    """
    Esta función verifica si cada palabra está presente en el vocabulario 
    del modelo Word2Vec y obtiene su vector correspondiente. 
    Luego, calcula el promedio de los vectores de palabras para obtener 
    el vector representativo de la lista de palabras.
    """
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def get_average_sentence_vector(sentences, word2vec_model):
    """
    Toma una lista de oraciones y aplica la función get_sentence_vector() 
    a cada oración para obtener los vectores de palabras promedio. 
    Luego, se realiza el promedio de estos vectores de oraciones para 
    obtener un vector de oración promedio
    """
    sentence_vectors = [get_sentence_vector(sentence, word2vec_model) for sentence in sentences]
    if sentence_vectors:
        return np.mean(sentence_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def embed_data_with_sentence_vector(sentences, word2vec_model):
    """
    Recorre las oraciones de cada muestra y obtiene el vector de oración promedio 
    utilizando get_average_sentence_vector(). Solo se agrega el vector de 
    oración promedio si no es un vector de ceros.
    """
    embedded_mean = []
    
    for sentence in sentences:
        sentence_vector = get_average_sentence_vector(sentence, word2vec_model)
        if np.any(sentence_vector):
            embedded_mean.append(sentence_vector)
    
    return embedded_mean

def word2vec_text_vectorization_embed(df):
    df = df.copy()

    # Entrenar el modelo Word2Vec
    sentences = df['lemmatize_word_pos']
    word2vec_model = train_word2vec_model(sentences)
    # Obtener los vectores promedio de las frases
    embedded_mean = embed_data_with_sentence_vector(sentences, word2vec_model)
    
    return word2vec_model, embedded_mean

# Preprocessing the other features
def get_additional_features(df, word2vec_model, embedded_mean):
    """
    Combinamos embedded_mean con las features adicionales, como las columnas 
    de desplazamiento (Pronoun-offset, A-offset y B-offset), 
    y las características codificadas de los nombres A y B (A y B) y Pronoun.
    """
    df = df.copy()
    
    # Obtener las características adicionales
    pronoun_offset = np.array(df['Pronoun-offset'])
    a_offset = np.array(df['A-offset'])
    b_offset = np.array(df['B-offset'])

    # Convertir las características adicionales en vectores de desplazamiento normalizados
    normalized_pronoun_offset = pronoun_offset / len(embedded_mean)
    normalized_a_offset = a_offset / len(embedded_mean)
    normalized_b_offset = b_offset / len(embedded_mean)

    # Obtener las representaciones vectoriales para Pronoun, A y B (si están disponibles)
    pronoun_vector = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word in df['Pronoun']]
    a_vector = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word in df['A']]
    b_vector = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word in df['B']]

    # Combinar las características en una matriz de características
    X = np.column_stack((embedded_mean, normalized_pronoun_offset, normalized_a_offset, normalized_b_offset, a_vector, b_vector))
    
    return X

# Converting target to binary
def encode_target(df, target_1="A-coref", target_2="B-coref"):
    """
    Debemos codificar las etiquetas de clase de las variables objetivo 
    (A-coref y B-coref) para asegurar la compatibilidad con los modelos.
    Dabido a que tenemos clases binarias, usaremos el LabelEncoder:
        - False = 0
        - True = 1
    """
    le = preprocessing.LabelEncoder()
    df[target_1] = le.fit_transform(df[target_1])
    df[target_2] = le.fit_transform(df[target_2])

    return df

# CREAR COLUMNA TARGET DEFINITIVA
# Añadir 3a condición: pronombre no hace referencia a ninguno de los nombres propuestos
def add_none_coref_column(df):
    """
    Puede haber casos en los que la respuesta sea A-coref == False y B-coref == False, 
    lo que indica que el pronombre no hace referencia a ninguno de los nombres propuestos. 
    """
    # Obtener las columnas A-coref y B-coref del dataset de entrenamiento
    a_coref = df['A-coref'].values
    b_coref = df['B-coref'].values

    # Crear la tercera columna para representar la clase "None"
    none_coref = np.where((a_coref == 0) & (b_coref == 0), 1, 0)

    # Agregar la columna "None-coref" al dataset de entrenamiento
    df['None-coref'] = none_coref

    return df

# Crear la columna target
def get_target(row):
    """
    Creamos la columna target definitiva donde asignamos:
    - `0`: si el pronombre hace referencia al nombre A.
    - `1`: si el pronombre hace referencia al nombre B.
    - `2`: si el pronombre no hace referencia a ninguno de los dos nombres propuestos. 
    """
    if row['A-coref'] == 1:
        return 0
    elif row['B-coref'] == 1:
        return 1
    else:
        return 2

def multiclass_target(df):
    """
    Crea la columna target definitiva.
    Agrega la columna "None-coref" y "target" al df, 
    donde la columna "target" contendrá los valores 0, 1 o 2 
    dependiendo de si el pronombre hace referencia al nombre A, B 
    o a ninguno de los dos nombres propuestos.
    """
    df = df.copy()
    # Añadir la columna None-coref al df
    df = add_none_coref_column(df)
    # Crear la columna target 
    df['target'] = df.apply(get_target, axis=1)

    return df

# Preprocess whole dataset using the previous functions
def preprocess_dataset(df):
    """
    Realiza el preprocesamiento completo del dataset.
    Devuelve las características X y los valores objetivo y.
    """
    
    # Text Cleaning
    df_cleaned = text_cleaning(df)

    # Text Preprocessing
    df_clean_prep = text_preprocessing(df_cleaned)

    # Text Features Extraction
    word2vec_model, embedded_mean = word2vec_text_vectorization_embed(df_clean_prep)

    # Preprocessing the other features
    # Build the X (features)
    X = get_additional_features(df_clean_prep, 
                                word2vec_model,
                                embedded_mean)

    # Converting target to binary
    df_clean_prep = encode_target(df_clean_prep, 
                                  target_1="A-coref", 
                                  target_2="B-coref")

    # Creamos la columna target definitiva
    df_final = multiclass_target(df_clean_prep)

    # y (target)
    y = df_final['target'].values.astype(int)
    
    return X, y

