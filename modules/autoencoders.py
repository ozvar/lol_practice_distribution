from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from tensorflow.keras.layers import Activation, GaussianNoise


class RecurrentAutoEncoder:
    """
    """
    def __init__(self, X, noise=0.2, units=50, latent_space=25, loss='mae',
                 optimizer='rmsprop', output_activation='relu'):
        """
        """
        self.__build_model(
            X=X,
            noise=noise,
            units=units,
            latent_space=latent_space,
            loss=loss,
            optimizer=optimizer,
            output_activation=output_activation
        )

    def __build_model(self, X, noise, units, latent_space, loss, optimizer,
                      output_activation):
        """
        """
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        corrupted_input = GaussianNoise(noise)(inputs)

        # encoder layers
        encoder = LSTM(units, return_sequences=True)(corrupted_input)

        # latent space
        encoder = LSTM(latent_space, name='latent_space')(encoder)
        latent_space = RepeatVector(X.shape[1])(encoder)

        # decoder layers
        decoder = LSTM(units, return_sequences=True)(latent_space)
        decoder = Dense(X.shape[2])(decoder)
        decoder = Activation(output_activation)(decoder)

        # our entire encoder decoder model
        encoder_decoder_model = Model(inputs, decoder)
        encoder_decoder_model.compile(loss=loss, optimizer=optimizer)

        setattr(self, '_encoder_decoder', encoder_decoder_model)
        setattr(self, '_encoder', Model(inputs, encoder))

    def fit(self, X, **kwargs):
        """
        """
        history = self._encoder_decoder.fit(
            X,
            X,
            **kwargs
        )
        return history

    def encode(self, X, **kwargs):
        """
        """
        encoding = self._encoder.predict(
            X,
            **kwargs
        )
        return encoding
