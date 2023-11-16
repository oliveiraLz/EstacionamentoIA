import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Verificar a disponibilidade da GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device = "/GPU:0"
else:
    device = "/CPU:0"
print(f"Dispositivo de Treinamento: {device}")

# Obter o caminho absoluto para a pasta do projeto
projeto_path = os.path.abspath(os.path.dirname(__file__))

# Carregar o conjunto de dados usando ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen = datagen.flow_from_directory(
    os.path.join(projeto_path, 'dataset'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = datagen.flow_from_directory(
    os.path.join(projeto_path, 'dataset'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Definir um modelo para classificação com quatro classes
model = models.Sequential()
resnet = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(224, 224, 3), pooling=None)
resnet.trainable = False  # Impede que os pesos da ResNet50 sejam treinados
model.add(resnet)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(4, activation='softmax'))

# Compilar o modelo
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento
num_epochs = 10

# Estrutura para armazenar histórico de perda
train_loss_history = []

for epoch in range(num_epochs):
    print(f'Início da Época {epoch + 1}')
    start_time = time.time()  # Registrar o tempo de início da época

    model.fit(train_datagen, epochs=1, steps_per_epoch=len(train_datagen))

    end_time = time.time()  # Registrar o tempo de término da época
    epoch_time = end_time - start_time
    print(f'Fim da Época {epoch + 1}, Tempo: {epoch_time:.2f} segundos')

# Avaliação no conjunto de teste
accuracy = model.evaluate(test_datagen)[1]
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')
