
import tensorflow as tf
import datetime, os

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(logdir)

model = tf.keras.applications.ResNet50()  # ⟵ your model

@tf.function  # ensure a graph is built
def forward(x):
    return model(x)

tf.summary.trace_on(graph=True, profiler=False)
_ = forward(tf.random.uniform([1, 224, 224, 3]))  # ⟵ example input

with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0)
