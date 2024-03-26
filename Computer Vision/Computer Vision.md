# Convolutional Layers

# Pooling Layers

El objetivo de estas capas son submuestrear la imagen de entrada para reducir:
- La carga computacional.
- El uso de la memoria.
- El número de parámetros.

Funciona de forma similar a las capas convolucionales. Se pasa un kernel sobre la imagen de entrada, solo que en estas no se calcula la convolución sino una operación matemática (generalmente el máximo entre los pixeles que ocupa el kernel en la imagen, o el promedio de estos).

# Necesidad de capas de ReLU

# Arquitecturas CNNs

## LeNet-5

Características

- Usado ampliamente en el dataset MNIST. Las imágenes de este dataset son de 28x28 pero se hicieron de 32x32 realizando la técnica de zero padding.
- El resto de la red no utiliza ninguna técnica de padding
- Las capas de pooling son un poco más compejas que las usuales. Realiza la media de los pixeles y luego multiplica el resultado por un coeficiente de aprendizaje (uno por mapa) y le adiciona el bias (uno por mapa), y finalmente aplica la función de activación.
- Cada neurona de la salida de la red calcula la distancia euclidiana entre el vector de entrada y el vector de peso, en vez de calcular la multiplicación entre ambos vectores.
- Se recomienda utilizar cross-entropy como función de costo para penalizar mucho más las decisiones malas, produciendo así gradientes grandes y convergiendo rápidamente.

**Arquitectura**

| Tipo | Feature Maps | Tamaño | Tamaño del kernel | Stride | Función de activación |
|--|--|--|--|--|--|
|Input||1|32x32|-|-|-|
|Convolutional|6|28x28|5x5|1|tanh|
|Average pooling|6|14x14|2x2|2|tanh|
|Convolucional|16|10x10|5x5|1|tanh|
|Average pooling|16|5x5|2x2|2|tanh|
|Convolucional|120|1x1|5x5|1|tanh|
|Fully connected|84|-|-|-|tanh|
|Fully connected|10|-|-|-|RBF|

## AlexNet

Es similar a la LeNet-5, solo que más profunda y grande. Y además realiza el stacking de varias capas convolucionales antes de aplicar pooling.

Características:

- Se usó para la competencia de ImageNet.
- Para el overfitting el autor utilizó dos técnicas de regularización:
  - Aplicó capas de Dropout del 50% durante el entrenamiento en las capas Fully Connected
  - Realizó Data Augmentation haciendo shifting aleatoriamente en las imágenes de entrenamiento con varios offsets, también las volteó horizontalmente, y cambió las condiciones de luz.
- Además realizó un tipo de normalización a la salida de las capas convolucionales 1 y 3, a la cual llamó response normalization (LRN). En tensorflow puede ser usada utilizando tf.nn.local_response_normalization(), y si quieres utilizar Keras necesitas aplicar una capa Lambda. En AlexNet se utilizaron los parámetros r=2, &alpha;=0.00002, &beta;=0.75, y k=1.
- Existe una variante de AlexNet llamada ZF Net, es similar solo que se realiza hypertunning sobre diferentes parámetros de su arquitectura.

| Tipo | Feature Maps | Tamaño | Tamaño del kernel | Stride | Padding |Función de activación |
|--|--|--|--|--|--|--|
|Input||3(RGB)|227x227|-|-|-|-|
|Convolutional|96|55x55|11x11|4|valid|ReLU|
|Max pooling|96|27x27|3x3|2|valid|-|
|Convolucional|256|27x27|5x5|1|same|ReLU|
|Max pooling|256|13x13|3x3|2|valid|-|
|Convolucional|384|13x13|3x3|1|same|ReLU|
|Convolucional|384|13x13|3x3|1|same|ReLU|
|Convolucional|256|13x13|3x3|1|same|ReLU|
|Max pooling|256|6x6|3x3|2|valid|-|
|Fully connected|4096|-|-|-|-|ReLU|
|Fully connected|4096|-|-|-|-|ReLU|
|Fully connected|1000|-|-|-|-|Softmax|

## GoogLeNet

- Fue la arquitectura ganadora para la competencia de ImageNet.
- Su buen desempeño proviene de utilizar una red más profunda que la CNN anterior. La profundidad fue realizada gracias a un modulo llamado inception modules que hace subredes. Esto posibilito el uso de parámetros de forma eficiente.

**Arquitertura de inception module**
PENDIENTE A IMAGEN

Puedes pensar al inception module como una capa convolucional potenciada, capaz de capturar patrones compejos.

**Arquitectura de GoogleNet**
PENDIENTE A IMAGEN

## VGGNet

## ResNet

## Xception

## SENet





