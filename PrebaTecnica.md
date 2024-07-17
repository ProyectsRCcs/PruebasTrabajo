# Análisis de Sentimiento en Redes Sociales
Este análisis explora los sentimientos expresados en un conjunto de datos de redes sociales, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático con BERT.
## Parte 1: Análisis Exploratorio
### Importación de Librerías
    import unicodedata
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import seaborn as sns
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords

### Descargar recursos de NLTK
    nltk.download('punkt')
    nltk.download('stopwords')

# Configuración y Carga de Datos
### Ruta del archivo CSV (ajusta según tu configuración)
    ruta_archivo = "PruebaTec.csv"
    columna_texto = 'message'

### Leer el archivo CSV
    df = pd.read_csv(ruta_archivo, delimiter=';', encoding='latin-1', on_bad_lines='skip')

### Verificar si la columna existe
    if columna_texto not in df.columns:
        raise KeyError(f"La columna '{columna_texto}' no existe en el archivo CSV.")

### Cargar stopwords en español y agregar palabras vacías adicionales
    stop_words = set(stopwords.words('spanish'))
    stop_words.update(['aa', 'aaa', 'aaaa', 'aaae', 'eeea', 'eaaa', 'aa12aa', 'aa1aa', 'a12a', 'taao', 'aaaaa', 'aaaaaa', 'eaaaaa', 'aaaaaaa', 'asaa', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía'])
### Preprocesamiento de Texto
    def preprocesar_texto(texto):
        # Normalizar a NFKD (descompone caracteres acentuados)
        texto = unicodedata.normalize('NFKD', texto)

        # Eliminar caracteres no deseados
        texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑäëïöüÄËÏÖÜ]', '', texto)

        # Convertir a minúsculas
        texto = texto.lower()

        # Tokenización (separar en palabras)
        tokens = nltk.word_tokenize(texto)

        # Eliminar stop words en español y palabras muy cortas
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words and len(token) > 2]

        # Unir los tokens de nuevo en una cadena
        texto_procesado = ' '.join(tokens)

        return texto_procesado

### Aplicar la función a la columna de texto
    df[columna_texto] = df[columna_texto].astype(str).apply(preprocesar_texto)

# Análisis y Visualizaciones
### Frecuencia de sentimientos
    frecuencia_sentimientos = df['sentiment'].value_counts()
    print("Frecuencia de Sentimientos:")
    print(frecuencia_sentimientos)

### Longitud de los mensajes
    df['longitud_texto'] = df['message'].apply(len)
    longitud_texto = df['longitud_texto'].describe()
    print("\nLongitud de los Mensajes:")
    print(longitud_texto)

### Cantidad de usuarios 
    if 'username' in df.columns:
        cantidad_usuarios = df['username'].nunique()
        print("\nCantidad de Usuarios:")
        print(cantidad_usuarios)

### Frecuencia de palabras
    all_words = ' '.join(df['message']).split()
    frecuencia_palabras = pd.Series(all_words).value_counts().head(20)
    print("\nFrecuencia de las Palabras Más Comunes:")
    print(frecuencia_palabras)

### Nube de palabras por sentimiento
    def generar_nube_palabras(df, sentimiento):
        texto = " ".join(df[df['sentiment'] == sentimiento]['message'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texto)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nube de palabras: {sentimiento}")
        plt.show()

    for sentimiento in df['sentiment'].unique():
        generar_nube_palabras(df, sentimiento)

### Histograma de longitud de texto por sentimiento
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='longitud_texto', hue='sentiment', multiple='stack')
    plt.title('Histograma de longitud de texto por sentimiento')
    plt.xlabel('Longitud del texto')
    plt.ylabel('Frecuencia')
    plt.show()

### Histograma de sentimientos por fecha
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Eliminar filas con fechas no válidas
    plt.figure(figsize=(12, 8))
    ax = sns.histplot(data=df, x='date', hue='sentiment', multiple='stack', shrink=0.8)
    plt.title('Histograma de sentimientos por fecha')
    plt.xlabel('Fecha')
    plt.ylabel('Frecuencia')

### Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    plt.show()

### Gráfico de barras de frecuencia de emojis
    df['tiene_emoji'] = df['message'].apply(lambda x: any(char in x for char in '😂🤣😊😍😭😡😱👍❤️'))
    conteo_emojis = df['tiene_emoji'].value_counts()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=conteo_emojis.index, y=conteo_emojis.values)
    plt.title('Frecuencia de mensajes con emojis')
    plt.xlabel('Contiene emoji')
    plt.ylabel('Cantidad de mensajes')
    plt.xticks([0, 1], ['No', 'Sí'])

### Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

### Gráfico de barras de frecuencia de días de la semana
    df['weekday'] = df['date'].dt.day_name()
    conteo_dias = df['weekday'].value_counts()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=conteo_dias.index, y=conteo_dias.values)
    plt.title('Frecuencia de mensajes por día de la semana')
    plt.xlabel('Día de la semana')
    plt.ylabel('Cantidad de mensajes')

### Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

## Parte 2: Clasificación de Sentimientos con BERT

La mejor elección depende de factores como el tamaño y la naturaleza de tu conjunto de datos, los recursos computacionales, la velocidad de inferencia requerida y los matices específicos de tu tarea de análisis de sentimiento. A menudo es una buena práctica experimentar con múltiples enfoques y comparar su rendimiento.
Consieraciones especificas por las cuales se escogió BERT: 
1. Comprensión contextual: BERT entiende el contexto de las palabras en una oración, lo cual es crucial para el análisis de sentimiento donde el significado de las palabras puede cambiar según su entorno.
2. Pre-entrenamiento: BERT está pre-entrenado en un gran corpus de texto, lo que le permite tener una sólida comprensión base del lenguaje antes de ajustarse a tareas específicas.
3. Capacidades multilingües: Existen modelos BERT pre-entrenados para muchos idiomas, incluyendo el español como se usa en este proyecto.
4. Rendimiento de vanguardia: BERT y sus variantes han logrado resultados superiores en muchos puntos de referencia de PNL, incluyendo tareas de análisis de sentimiento.
5. Manejo de dependencias de largo alcance: La arquitectura de transformadores permite a BERT capturar relaciones entre palabras que están alejadas en el texto.

### Importación de Librerías
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from transformers import BertForSequenceClassification, BertTokenizer, AdamW
    from torch.utils.data import DataLoader, TensorDataset
    import torch

### Cargar datos
    column_names = ['number', '_id', 'message', 'name', 'id_user', 'username', 'id_post', 'link', 'date', 'user_link', 'weekday', 'just_emoji', 'sentiment', 'reply_screen_name', 'created_at', 'owner', 'shortcode', 'hour'] 
    data = pd.read_csv("PruebaTec.csv", names=column_names, on_bad_lines='skip')
    data.dropna(subset=["sentiment"], inplace=True)

### Preprocesar datos
    X = data['message'].astype(str).values
    y = data['sentiment'].values

### Convertir etiquetas de sentimiento a valores numéricos
    unique_sentiments = data['sentiment'].unique()
    label_map = {sentiment: i for i, sentiment in enumerate(unique_sentiments)}
    y = np.array([label_map[sentiment] for sentiment in y])

### Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Cargar tokenizador y modelo BERT pre-entrenado en español
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=len(unique_sentiments))

### Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

### Hiperparámetros
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    def tokenize_data(texts, labels, tokenizer, max_length=128):
        encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, labels

### Crear dataloaders
    train_input_ids, train_attention_mask, train_labels = tokenize_data(X_train, y_train, tokenizer)
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Optimizador
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

### Entrenamiento
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} completed")

### Validación
    model.eval()

### Tokenizar datos de prueba
    test_input_ids, test_attention_mask, test_labels = tokenize_data(X_test, y_test, tokenizer)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

### Evaluación del modelo
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

### Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

### Guardar el modelo
    torch.save(model.state_dict(), 'sentiment_model.pth')
    print("Modelo guardado como 'sentiment_model.pth'")

# Interpretación y Conclusiones
El modelo BERT muestra un rendimiento prometedor (75% de precisión) en la tarea de clasificación de sentimientos, pero se necesitan más datos y experimentos para una evaluación más robusta.
# Limitaciones:
- El conjunto de datos es pequeño y puede no ser representativo.
- La precisión del 75% sugiere que hay margen de mejora.

# Recomendaciones:

1. Ampliar el conjunto de datos para mejorar la generalización del modelo.
2. Experimentar con diferentes hiperparámetros (tasa de aprendizaje, número de épocas, tamaño de lote).
3. Considerar técnicas de aumento de datos para clases minoritarias.
4. Evaluar el modelo en un conjunto de datos de prueba independiente y real.
5. Implementar validación cruzada para una evaluación más robusta.
6. Analizar los errores de clasificación para identificar patrones y áreas de mejora.
7. Realizar un análisis de errores detallado.
8. Experimentar con diferentes arquitecturas de modelo (por ejemplo, RoBERTa, XLM-RoBERTa).
9. Implementar técnicas de interpretabilidad del modelo para entender mejor sus decisiones.
10. Considerar el uso de técnicas de aprendizaje por transferencia con ajuste fino en dominios específicos.
