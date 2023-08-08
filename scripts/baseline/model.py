import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Masking, Bidirectional, LSTM, Layer, Flatten, Concatenate

class Baseline(tf.Module):
    '''
    Baseline Bidirectional LSTM module
    '''
    def __init__(self, n_units=32, rec_layers=1, sigmoid=True, name='Baseline'):
        ''' Baseline BiLSTM initialization

        Args:
            n_units: (int) Number of recurrent units
            rec_layers: (int) Number of recurrent layers to stack upon each other
            sigmoid: (bool) Add Sigmoid activation layer at output
            name: (str) Module name
        '''
        super().__init__(name=name)

        ## LAYERS ##
        self.mask = Masking(mask_value=-1.)

        # Set return_sequences=True in first layers of BiLSTM stacked layers
        self.rec = [Bidirectional(LSTM(n_units, return_sequences=True)) for _ in range(rec_layers-1)]
        self.rec.append(Bidirectional(LSTM(n_units)))

        # Output layer 
        if sigmoid:
            self.out = Dense(1, activation='sigmoid')
        else:
            self.out = Layer() # Identity pass through
    
    def __call__(self, input_):
        '''Baseline forward pass'''
        # Mask input
        x = self.mask(input_)
        # Pass through stacked BiLSTM layers
        for rec_layer in self.rec:
            x = rec_layer(x)
        return self.out(x)

def baseline_model(n_units=32, rec_layers=1):
    '''Baseline model (BiLSTM)
    
    Args:
        n_units: Number of recurrent units, default = 32
        rec_layers: Number of stacked recurrent layers, default = 1

    Returns:
        BiLSTM baseline model
    '''
    model = Sequential()
    model.add(Baseline(n_units, rec_layers))
    return model
