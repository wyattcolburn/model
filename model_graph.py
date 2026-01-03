import visualkeras
from tensorflow.keras.models import load_model

cnn = load_model("data_set/2025_08_31_14_59/2025_08_31_14_59.keras")
visualkeras.layered_view(cnn, to_file="mlp_arch.png", legend=True)
