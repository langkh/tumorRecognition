import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from matplotlib import pyplot
from matplotlib.image import imread

batch_size= 32
img_height =180
img_width =180

train_ds = tf.keras.utils.image_dataset_from_directory(
	r"C:\Users\schoo\Tumor Recognition\images\Training",#copy path name here
	validation_split = 0.2,
	subset = "training",
	seed = 123,
	image_size = (img_height, img_width),
	batch_size = batch_size
	)
val_ds = tf.keras.utils.image_dataset_from_directory(
	r"C:\Users\schoo\Tumor Recognition\images\Testing",#copy path name here
	validation_split = 0.2,
	subset = "validation",
	seed = 123,
	image_size = (img_height, img_width),
	batch_size = batch_size
	)
class_names = train_ds.class_names
print(class_names)

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

model= Sequential([
	layers.Rescaling(1./255, input_shape = (img_height,img_width, 3)),
	layers.Conv2D(16,3,padding = "same",activation = 'relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(16,3,padding = "same",activation = 'relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(16,3,padding = "same",activation = 'relu'),
	layers.Flatten(),
	layers.Dense(128,activation = "relu"),
	layers.Dense(num_classes)


		])

model.compile(optimizer = "adam",
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics = ['accuracy'])
model.summary()

epochs = 15
history = model.fit(
	train_ds,
	validation_data = val_ds,
	epochs = epochs
)
img = tf.keras.utils.load_img(
	r"C:\Users\schoo\Tumor Recognition\images\Testing\no_tumor\image(2).jpg",target_size = (180,180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array,0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(class_names[np.argmax(score)])
