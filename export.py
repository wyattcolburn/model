
# in your TF 2.20 / Keras 3 env
import tensorflow as tf

keras_path = "data_set/2025_08_31_14_59/2025_08_31_14_59.keras"
sm_path    = "data_set/2025_08_31_14_59/savedmodel"

model = tf.keras.models.load_model(keras_path)
tf.saved_model.save(model, sm_path)            # export as TF SavedModel
# (equivalently: model.save(sm_path, save_format="tf"))
