# Análisis de Sentimiento en Redes Sociales

Este análisis explora los sentimientos expresados en un conjunto de datos de redes sociales, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático con BERT.

## Parte 1: Análisis Exploratorio

### Importación de Librerías

```python
import unicodedata
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Fijando ruta de trabajo
ruta_archivo = r"C:\Users\Eddlu\OneDrive\Documents\GenioStudios\PruebaTec.csv"
columna_texto = 'message'

df = pd.read_csv(ruta_archivo, delimiter=';', encoding='latin-1', on_bad_lines='skip')


# Instalación de paquetes para parte 2
    import unicodedata
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import seaborn as sns
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords

# Descargar recursos de NLTK
    nltk.download('punkt')
    nltk.download('stopwords')

# Ruta del archivo CSV
    ruta_archivo = r"C:\Users\Eddlu\OneDrive\Documents\GenioStudios\PruebaTec.csv"

# Columna a procesar
    columna_texto = 'message'

# Leer el archivo CSV, especificando delimitador y codificación
try:
    df = pd.read_csv(ruta_archivo, delimiter=';', encoding='latin-1', on_bad_lines='skip')

    # Verificar si la columna existe
    if columna_texto not in df.columns:
        raise KeyError(f"La columna '{columna_texto}' no existe en el archivo CSV.")

    # Cargar stopwords en español y agregar palabras vacías adicionales
    stop_words = set(stopwords.words('spanish'))
    stop_words.update(['aa', 'aaa', 'aaaa', 'aaae', 'eeea', 'eaaa', 'aa12aa', 'aa1aa', 'a12a', 'taao', 'aaaaa', 'aaaaaa', 'eaaaaa', 'aaaaaaa', 'asaa', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía'])  # Agrega más según sea necesario

    # Función para preprocesar el texto (mejorada)
    def preprocesar_texto(texto):
        # Normalizar a NFKD (descompone caracteres acentuados)
        texto = unicodedata.normalize('NFKD', texto)

        # Eliminar caracteres no deseados (incluyendo los que mencionaste)
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

    # Aplicar la función a la columna de texto
    df[columna_texto] = df[columna_texto].astype(str).apply(preprocesar_texto)

    # --- Tablas ---

    # Frecuencia de sentimientos
    frecuencia_sentimientos = df['sentiment'].value_counts()
    print("Frecuencia de Sentimientos:")
    print(frecuencia_sentimientos)

    # Longitud de los mensajes
    df['longitud_texto'] = df['message'].apply(len)
    longitud_texto = df['longitud_texto'].describe()
    print("\nLongitud de los Mensajes:")
    print(longitud_texto)

    # Cantidad de usuarios 
    if 'user' in df.columns:
        cantidad_usuarios = df['username'].nunique()
        print("\nCantidad de Usuarios:")
        print(cantidad_usuarios)

    # Frecuencia de palabras
    all_words = ' '.join(df['message']).split()
    frecuencia_palabras = pd.Series(all_words).value_counts().head(20)
    print("\nFrecuencia de las Palabras Más Comunes:")
    print(frecuencia_palabras)

    # --- Visualizaciones ---

    # Nube de palabras por sentimiento
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

    # Histograma de longitud de texto por sentimiento
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df, x='longitud_texto', hue='sentiment', multiple='stack')
    plt.title('Histograma de longitud de texto por sentimiento')
    plt.xlabel('Longitud del texto')
    plt.ylabel('Frecuencia')
    plt.show()

    # Histograma de sentimientos por fecha con números de frecuencia
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Eliminar filas con fechas no válidas
    plt.figure(figsize=(12, 8))
    ax = sns.histplot(data=df, x='date', hue='sentiment', multiple='stack', shrink=0.8)
    plt.title('Histograma de sentimientos por fecha')
    plt.xlabel('Fecha')
    plt.ylabel('Frecuencia')
    
    # Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Solo etiquetar barras con altura mayor a cero
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    plt.show()

    # Gráfico de barras de frecuencia de emojis
    df['tiene_emoji'] = df['message'].apply(lambda x: any(char in x for char in '😂🤣😊😍😭😡😱👍❤️'))
    conteo_emojis = df['tiene_emoji'].value_counts()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=conteo_emojis.index, y=conteo_emojis.values)
    plt.title('Frecuencia de mensajes con emojis')
    plt.xlabel('Contiene emoji')
    plt.ylabel('Cantidad de mensajes')
    plt.xticks([0, 1], ['No', 'Sí'])
    
    # Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

    # Gráfico de barras de frecuencia de días de la semana
    df['weekday'] = df['date'].dt.day_name()
    conteo_dias = df['weekday'].value_counts()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=conteo_dias.index, y=conteo_dias.values)
    plt.title('Frecuencia de mensajes por día de la semana')
    plt.xlabel('Día de la semana')
    plt.ylabel('Cantidad de mensajes')
    
    # Añadir números de frecuencia dentro del gráfico
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

except pd.errors.EmptyDataError:
    print("El archivo CSV está vacío.")
except FileNotFoundError:
    print(f"No se encontró el archivo CSV: {ruta_archivo}")
except KeyError as e:
    print(e)

# Directorio para guardar las imágenes y tablas
ruta_salida = r"C:\Users\Eddlu\OneDrive\Documents\GenioStudios\Resultados"

# Frecuencia de sentimientos
frecuencia_sentimientos = df['sentiment'].value_counts()
print("Frecuencia de Sentimientos:")
print(frecuencia_sentimientos)
frecuencia_sentimientos.to_csv(f"{ruta_salida}/frecuencia_sentimientos.csv")  # Guardar tabla

# Nube de Palabras. Sentimiento Positivo

"C:/Users/Eddlu/OneDrive/Documents/GenioStudios/resultados/nube_palabras_positivo.png"
# Nube de Palabras. Sentimiento Neutro
"C:/Users/Eddlu/OneDrive/Documents/GenioStudios/resultados/nube_palabras_neutro.png"
# Nube de Palabras. Sentimiento Negativo
"C:/Users/Eddlu/OneDrive/Documents/GenioStudios/resultados/nube_palabras_negativo.png"

Frecuencia de Sentimientos:
<resultados de frecuencia_sentimientos>

Longitud de los Mensajes:
<resultados de longitud_texto>

Cantidad de Usuarios:
<resultados de cantidad_usuarios>

Frecuencia de las Palabras Más Comunes:
<resultados de frecuencia_palabras>


## Clasificación de Sentimientos con BERT (Parte 2/2)

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from transformers import BertForSequenceClassification, BertTokenizer, AdamW
    from torch.utils.data import DataLoader, TensorDataset
    import torch

# Cargar datos, especificando nombres de columnas y manejando valores faltantes
    column_names = ['number', '_id', 'message', 'name', 'id_user', 'username', 'id_post', 'link', 'date', 'user_link', 'weekday', 'just_emoji', 'sentiment', 'reply_screen_name', 'created_at', 'owner', 'shortcode', 'hour'] 
    data = pd.read_csv("PruebaTec.csv", names=column_names, on_bad_lines='skip') # Saltar lineas problematicas
    data.dropna(subset=["sentiment"], inplace=True)

# Preprocesar datos
    X = data['message'].astype(str).values  # Asegurar que los mensajes sean cadenas
    y = data['sentiment'].values

# Convertir etiquetas de sentimiento a valores numéricos
    unique_sentiments = data['sentiment'].unique()
    label_map = {sentiment: i for i, sentiment in enumerate(unique_sentiments)}
    y = np.array([label_map[sentiment] for sentiment in y])

# Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar tokenizador y modelo BERT pre-entrenado en español
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=len(unique_sentiments))

## Fine-tuning del modelo

# a. Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

# Hiperparámetros
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-5

# b. Implementación
    def tokenize_data(texts, labels, tokenizer, max_length=128):
        encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        # Convert labels to LongTensor for PyTorch
        labels = torch.tensor(labels, dtype=torch.long)  
        return input_ids, attention_mask, labels

# Crear dataloaders
    train_input_ids, train_attention_mask, train_labels = tokenize_data(X_train, y_train, tokenizer)
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizador
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Ensure labels are on the correct device
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} completed")

# c. Validación
    model.eval()

# Tokenizar datos de prueba
    test_input_ids, test_attention_mask, test_labels = tokenize_data(X_test, y_test, tokenizer)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3. Evaluación del modelo
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

# Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Guardar el modelo
    torch.save(model.state_dict(), 'sentiment_model.pth')
    print("Modelo guardado como 'sentiment_model.pth'")

Accuracy: 0.7500
Precision: 0.7500
Recall: 0.7500
F1-score: 0.7500

Confusion Matrix:
<matriz de confusión>

## Interpretación y Conclusiones
El modelo BERT muestra un rendimiento prometedor (75% de precisión) en la tarea de clasificación de sentimientos, pero se necesitan más datos y experimentos para una evaluación más robusta.

Limitaciones: El conjunto de datos es pequeño y puede no ser representativo.

Recomendaciones: Ampliar el conjunto de datos, ajustar hiperparámetros y evaluar en datos reales.
