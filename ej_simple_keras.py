from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])



data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 10))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

loss_and_metrics= model.evaluate(data, labels, batch_size=128)

classes = model.predict(data, batch_size=128)
print(loss_and_metrics)
print(classes) 
