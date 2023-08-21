import tensorflow as tf

model = tf.keras.models.load_model('model\model_v3')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model/trained_model_v3.tflite", "wb").write(tflite_model)