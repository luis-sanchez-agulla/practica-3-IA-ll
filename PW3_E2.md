# 1. Preprocesamiento — Palabras frecuentes por clase

Con el fin de inspeccionar en mayor detalle cuáles son las palabras más frecuentes dentro de cada una de las clases que tiene nuestro corpus, buscamos visualizar a través de una gráfica de barras horizontales cuáles son dichas palabras.

---

# 2. Implementación

## 2.1. Extracción de Embeddings

Para poder realizar la construcción de un conjunto de modelos de procesamiento de lenguaje natural, independientemente de la arquitectura a seguir, se necesita de la incrustación de los *embeddings*, que son las representaciones vectoriales del texto como tal que compone el conjunto de datos o corpus.

La importación de los *embeddings* necesita de la extracción de estos datos desde una carpeta que se trae desde el primer ejercicio realizado dentro de la práctica. Para ello, importamos tanto los pesos como el vocabulario empleado desde dichas carpetas para poder asignar correctamente cada uno de los pesos a los valores numéricos extraídos del vocabulario elegido. Esto es necesario dado que el vocabulario está compuesto por un conjunto de palabras específicas, en un orden específico, y cada uno de los pesos se corresponde a un número o índice indicativo de la palabra a la que está relacionado.

Dentro del conjunto total de *embeddings* exportados del Ejercicio 1, buscamos utilizar dos en concreto, los cuales son:

* `embedding-dim312_v2`: esta incrustación nos permite observar el comportamiento de la red neuronal recurrente bajo la premisa de una ventana de contexto pequeña (tamaño de 2) y con una dimensionalidad reducida, de forma que puede servir como un buen punto de partida para poder analizar cuál es el comportamiento de nuestra red neuronal sin necesidad de emplear conjuntos de *embeddings* de dimensionalidad muy alta y ventanas de contexto más amplias.
* `embedding-dim752_v3`: tras la ejecución del primer ejercicio, este *embedding* es, de manera consistente, el que mejor métrica en la función de pérdida expone en comparación con el resto de incrustaciones potenciales resultantes del primer ejercicio. Por esto mismo, dado que esta incrustación tiene el potencial de poder usarse con una mejor base que el resto de las extraíbles del primer ejercicio, decidimos emplear esta.

## 2.2. Construcción de modelos

Con el fin de resolver el problema de clasificación de texto, buscamos crear un conjunto de redes neuronales recurrentes que nos permitan tratar el lenguaje natural desde diferentes perspectivas.

Un modelo de una red neuronal que cumple con estas condiciones necesita de diferentes componentes:

* **Capa de Incrustación (Embedding Layer):** empleamos la función `importar_embeddings` y la matriz pre-entrenada (`emb_matrix`), haciendo la traducción de los índices numéricos de las palabras a vectores densos que contienen su significado semántico.
* **Capas Recurrentes (LSTM o GRU):** dependiendo del parámetro `arq`, el modelo utiliza celdas de Memoria a Corto y Largo Plazo (LSTM) o Unidades Recurrentes Cerradas (GRU) para procesar la secuencia de palabras y capturar su contexto temporal. Dado que el número de capas dentro de este bloque puede ser cambiante, establecemos la entrada `num_capas` para determinar cuántas capas recurrentes necesitamos:
    * **Capas intermedias (`return_sequences=True`):** si hay más de una capa recurrente, las primeras deben devolver la secuencia completa de estados ocultos (un vector por cada palabra) para que la siguiente capa recurrente tenga una secuencia temporal que procesar.
    * **Capa final (`return_sequences=False` por defecto):** la última capa recurrente resume todo el contexto de la oración en un único vector denso final que representa el significado global del texto.
* **Mecanismos de Regularización Avanzada:** para evitar el potencial sobreajuste (*overfitting*) resultante de las capas recurrentes, el modelo implementa:
    * **`SpatialDropout1D` (0.2):** aplicado entre las capas recurrentes. A diferencia del Dropout tradicional, este apaga dimensiones enteras del vector a lo largo de toda la secuencia en lugar de unidades individuales, lo que fuerza a las capas siguientes a no depender de una sola característica lingüística. El empleo de esta capa nace de la necesidad de implementar *dropout* dentro del bloque recurrente sin perder la funcionalidad propia de CuDNN, que nos permite entrenar de manera mucho más eficaz en GPU estas redes neuronales.
    * **Regularización L2 (0.001):** aplicada en la primera capa densa. Penaliza matemáticamente los pesos demasiado grandes, obligando al modelo a aprender patrones más suaves y generales.
    * **`Dropout` estándar (0.5):** apaga aleatoriamente el 50% de las neuronas de la capa densa antes de la decisión final, forzando una votación más robusta.
* **Cabecera de Clasificación (Capas Densas):** una vez que el texto ha sido procesado por el núcleo recurrente, la información pasa a una capa `Dense` de 64 unidades con activación ReLU (para aprender combinaciones no lineales complejas), seguida por la capa final de salida. Esta última tiene tantas neuronas como categorías a predecir (`num_clases`) y utiliza la activación **`softmax`** para convertir los resultados en una distribución de probabilidades (ej. 80% Deportes, 15% Política, 5% Tecnología).
* **Configuración de Compilación:** por último, el modelo ensambla sus instrucciones de aprendizaje. Utiliza el optimizador **Adam** (que ajusta la tasa de aprendizaje definida por `lr`), y la función de pérdida **`categorical_crossentropy`**, que es el estándar matemático ideal para penalizar los errores en problemas de clasificación multiclase donde las etiquetas están codificadas en formato *One-Hot*.

## 2.3. Configuración de Callbacks

Una vez construida la arquitectura del modelo, es fundamental establecer mecanismos de control durante la fase de entrenamiento. Con el fin de combatir el fenómeno del *overfitting*, tratamos de implementar dos llamadas a funciones que nos permitan controlar la evolución del entrenamiento de los modelos.

Para gestionar esto de forma automática y óptima, la función `definir_callbacks` busca implementar dos monitoreos distintos sobre el modelo de red neuronal entrenado:

* **EarlyStopping:** detiene el entrenamiento en el momento que deja de mejorar su predicción tras varias iteraciones/épocas.
    * **`monitor='val_loss'`**: tomamos como referencia la función de pérdida en el conjunto de validación.
    * **`patience=3`**: damos un margen de mejora de 3 épocas consecutivas.
    * **`restore_best_weights=True`**: el modelo descarta los pesos de la última época y recupera automáticamente los pesos de la época en la que obtuvo su mejor rendimiento.
* **ReduceLROnPlateau:** a medida que el modelo se acerca a la solución óptima (el mínimo de la función de pérdida), dar "pasos" matemáticos muy grandes puede hacer que se salte dicha solución.
    * **`patience=1`**: es más impaciente que el EarlyStopping. Si la pérdida de validación se estanca durante una sola época, entra en acción.
    * **`factor=0.5`**: multiplica la tasa de aprendizaje (*Learning Rate*) actual por 0.5 (es decir, la reduce a la mitad). Esto obliga al modelo a dar pasos más pequeños y precisos, ayudándole a "acomodarse" mejor en los mínimos locales.
    * **`min_lr=1e-6`**: establece un límite para evitar que la tasa de aprendizaje se vuelva microscópica y el entrenamiento se congele por completo.

## 2.4. Entrenamiento de modelos

Una vez definidos los modelos, buscamos entrenar cada uno de ellos con el corpus procesado. Para ello, tratamos de realizar el entrenamiento de cada uno de los modelos de manera secuencial, comparando el rendimiento de los distintos modelos como tal. Una vez realizado el entrenamiento, buscamos quedarnos con el mejor resultado obtenido por cada una de las arquitecturas seleccionadas para resolver nuestro problema: LSTM y GRU.

Definimos el conjunto de configuraciones que vamos a probar de cada una de las arquitecturas seleccionadas para resolver nuestro problema. Buscamos probar diferentes valores dentro de los hiperparámetros como lo son el conjunto de unidades o neuronas añadidas dentro de las capas recurrentes propias de cada arquitectura, el número de capas recurrentes que vamos a necesitar dentro de nuestro modelo y la tasa de aprendizaje del optimizador Adam.

En concreto, evaluaremos las arquitecturas **LSTM** y **GRU** utilizando distintas matrices de *embeddings* (de dimensiones `312` y `752`). Las configuraciones varían en los siguientes hiperparámetros:

* **Unidades recurrentes:** modelos con `16`, `32`, `64` y `128` neuronas.
* **Número de capas:** principalmente modelos de `1` capa, pero también probaremos arquitecturas más profundas de `2` capas (con 32 unidades).
* **Tasa de aprendizaje (*Learning Rate*):** valores de `0.001` y un paso más conservador de `0.0001`.

Empleamos una táctica de entrenamiento de *mini-batch*, de forma que definimos el número de documentos que usamos en cada iteración para entrenar siendo este de `64` (`BATCH_CLF`), además de definir como máximo de épocas para el entrenamiento de nuestros modelos en `8` (`EPOCHS_CLF`).

Para poder evaluar correctamente el comportamiento de cada uno de los modelos a lo largo del entrenamiento, graficamos su evolución en función de las métricas propias que se usan en el entrenamiento: *accuracy* y *loss*.

De esta forma, resaltamos los mejores valores de cada una de las configuraciones propias que hemos definido, dado que cabe la posibilidad de que el mejor valor no corresponda al de la última época.

Podemos observar dentro de las gráficas de la evolución del *accuracy* y de la función de *loss* de cada uno de los modelos que el comportamiento general sigue una tendencia particular: los modelos que están empleando una tasa de aprendizaje más alta (por ejemplo, `lr=0.001`) tienden a un *overfitting* más pronunciado y rápido. Por ejemplo, el modelo `LSTM_d16_1L_lr001` experimentó un sobreajuste mayor con una diferencia final de *loss* (`diff_loss`) de `0.1054`, activando rápidamente los *callbacks*. En comparación, el conjunto de modelos que utilizan una tasa de aprendizaje menor (`lr=0.0001`) presentan un aprendizaje mucho más estable, como se ve en el modelo `LSTM_d16_1L_lr0001`, que logró un extraordinario `diff_loss` de apenas `0.0010` y un *accuracy* de validación de `0.8912`.

Por otro lado, el comportamiento de los modelos con respecto a su función de pérdida es tal que el conjunto de configuraciones propias de modelos más pequeños, como los que emplean `16` o `32` unidades, tienen un comportamiento evolutivo mucho más estable y convergente que el propio de modelos más grandes o con mayor cantidad de neuronas dentro de las capas recurrentes. Esto queda demostrado empíricamente al ver que los mejores modelos generales fueron arquitecturas compactas: el mejor modelo LSTM fue precisamente el de `16` unidades (`LSTM_d16_1L_lr0001` con `diff_loss=0.0010`), y el mejor modelo GRU fue el de `32` unidades (`GRU_d32_1L_lr0001` con `diff_loss=0.0023`).

Otra parte esencial de la evaluación de cada uno de los modelos es la inspección de la clasificación en particular de cada uno de los documentos dentro de los mejores modelos extraídos de las configuraciones dadas. De esta forma, podemos observar si existen patrones dentro de la clasificación errónea del modelo, entre otras posibilidades que habilita la visualización de la matriz de confusión.

Como es observable en la imagen anterior de las matrices de confusión, ambos modelos comparten un comportamiento sobre las predicciones realizadas bastante similar, denotando que el entrenamiento de ambos modelos, a pesar de las diferencias entre las dimensiones de las incrustaciones <!-- REVISAR: ambos modelos usan dim312_v2, no hay diferencia de dimensión entre ellos, al menos por lo que yo he visto, puede que se haya colado de alguna version anterior?? -->, el conjunto de neuronas y otros parámetros, es bastante similar. Esto puede ser indicativo de que la naturaleza propia del corpus introduce un ligero sesgo en ciertos documentos pertenecientes a clases específicas, lo cual hace que haya una leve tendencia a la confusión entre diferentes clases dentro del corpus.

---

# 3. Evaluación y preguntas

## 3.1. Evaluación con corpus ajeno

Para evaluar el funcionamiento de los dos modelos extraídos como los mejores de entre todos los posibles, procuramos emplear un corpus de noticias extraídas desde diferentes fuentes ajenas al corpus de entrenamiento, con el fin de preparar una batería de pruebas con la que podamos analizar en mejor perspectiva cómo se comporta cada uno de los modelos que hemos escogido, tanto el que emplea arquitectura `LSTM` como el que emplea arquitectura `GRU`.

La evaluación de nuestros modelos requiere que el texto contenido dentro del corpus de evaluación tenga el mismo tratamiento y preprocesado realizado que el propio corpus de entrenamiento. Es por ello que, tras importarlo, buscamos limpiar el texto de expresiones y palabras que no aporten información, así como de signos de puntuación y demás caracteres inservibles en el entrenamiento.

Una vez hemos procesado los datos del corpus de evaluación, podemos realizar un análisis de la propia predicción de cada uno de los modelos, de forma que podamos discernir entre ambos cuál es mejor realizando predicciones en un potencial proceso de inferencia.

## 3.2. Preguntas

**¿Ha alcanzado su modelo un entrenamiento óptimo? Justifique su respuesta con observaciones relevantes.**

Nuestros modelos escogidos de entre las posibles arquitecturas a usar para poder resolver el problema de clasificación de textos han tenido un entrenamiento óptimo. Esto es justificable a través de la observación de diferentes puntos sobre los que podemos basar nuestros argumentos:

* **Generalización:** nuestros modelos en ambas ocasiones tienen una buena generalización, dado que la diferencia entre el valor de la función de pérdida entre el entrenamiento y la validación es bastante reducida, siendo casos como el de `LSTM_d16_1L_lr0001`, el cual contiene una diferencia entre ambas métricas de tan solo 0,001. Esta indicación da a entender que el fenómeno del *overfitting* no se da prácticamente.
* **Convergencia estable:** ambos modelos seleccionados muestran un comportamiento estable a lo largo de las épocas de entrenamiento, dado que métricas como el `val_loss` descienden de manera uniforme, sin presentar saltos bruscos o cambios notables, y el *accuracy* también presenta una evolución consistente.

Para haber conseguido estos dos indicativos de un entrenamiento óptimo, se ha debido emplear dentro de la arquitectura de red neuronal una regularización aplicada a diferentes capas y de diferentes formas, como lo son el uso de L2, así como el uso de capas de `Dropout`.

**¿Qué revela la matriz de confusión sobre el rendimiento de su modelo? Analice la matriz para evaluar la capacidad de generalización de su modelo.**

La matriz de confusión revela un rendimiento muy sólido y consistente de ambos modelos. Observamos una fuerte concentración de predicciones en la diagonal principal, lo que indica que la gran mayoría de las muestras han sido clasificadas correctamente. Los valores fuera de la diagonal (falsos positivos y falsos negativos) son relativamente bajos y simétricos, indicando que el modelo no está excesivamente sesgado hacia una clase en particular. Podemos observar que sí se presenta una ligera tendencia a la confusión entre las clases 1, 3 y 4, teniendo la clase 2 una menor tasa de fallo en su predicción.

En cuanto a su capacidad de generalización, el comportamiento de la matriz de confusión sobre los datos de test, así como en el corpus extraído de manera independiente (custom), corrobora que el modelo ha aprendido características semánticas subyacentes útiles y no simplemente memorizado los datos. Las ligeras confusiones que presenta suelen estar justificadas por el solapamiento natural del vocabulario entre temáticas afines (por ejemplo, noticias empresariales vs. tecnología).

**¿Qué diferencia en la precisión final observó al cambiar de una representación de incrustación a otra?**

Al comparar el rendimiento entre las incrustaciones de menor dimensión (`dim312_v2`) y las de mayor dimensión (`dim752_v3`), se observa que el aumento de dimensionalidad no supuso una mejora significativa y estable en la precisión de validación, sino que más bien incrementó la tendencia al sobreajuste (*overfitting*). De hecho, los modelos que lograron la mejor relación entre *accuracy* y menor diferencia de pérdidas (`diff_loss`) fueron aquellos entrenados con los *embeddings* más compactos (`dim312_v2`). Esto se debe a que una mayor cantidad de dimensiones introduce una cantidad mucho mayor de parámetros en el modelo, lo que facilita que la red memorice ruido o características muy específicas del conjunto de entrenamiento en lugar de aprender representaciones lingüísticas generalizables a datos no vistos.

**¿La arquitectura LSTM/GRU reaccionó de forma diferente a los dos tipos de incrustación?**

En términos generales, ambas arquitecturas (LSTM y GRU) reaccionaron de forma muy similar a los dos tipos de incrustación. Ninguna de las dos obtuvo un beneficio claro de utilizar los *embeddings* de mayor dimensionalidad (`dim752_v3`). Por el contrario, tanto LSTM como GRU sufrieron un mayor sobreajuste al emplear las incrustaciones de 752 dimensiones, debido al drástico aumento en la cantidad de parámetros a entrenar. Como resultado, los modelos más óptimos y estables para ambas arquitecturas se obtuvieron utilizando las incrustaciones de menor dimensión (`dim312_v2`), demostrando que, para este corpus y problema específico, una representación más compacta es preferible independientemente del tipo de celda recurrente utilizada.
