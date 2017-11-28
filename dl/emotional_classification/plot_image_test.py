from keras.models import load_model
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz/bin/'

model = load_model('model/pn_test_all_1123.h5')
plot_model(model, to_file='model_tb.png', show_shapes=True, rankdir='TB')
plot_model(model, to_file='model_tb.png', show_shapes=True, rankdir='LR')
