# Architectures to get more robust results

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Reshape, Layer, Lambda, Concatenate, LayerNormalization, MultiHeadAttention, Add, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.optimizers import Adadelta
from keras.layers import BatchNormalization

# Base Model:
"""Simple Architecture:
 - Input 3D
 - LSTM layer 1 (return sequences), Batch_norm to handle covariate shifts, Dropout for regularization
 - LSTM layer 2 (returns single output), Batch_norm 2, Dropout 2 (half cell size of 1) 
 - Output Dense softmax for multiple up-down-sideways
 - Optimizer Adadelta
 - loss = sparse_categorical_crossentropy
 - metrics accuracy"""

def Base_Model_LSTM(X_train,X_valid,X_test,Y_train,Y_valid,Y_test, epochs=2, batch_size=100, d1=0.1, d2 = 0.05, cell_size = 80):
    # Clearing the TensorFlow session to ensure the model starts with fresh weights and biases
    tf.keras.backend.clear_session()
    n_classes = 3

    cell_size_1 = cell_size
    cell_size_2 = cell_size_1//2

    # Model definition
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    Lstm_layer_1 = LSTM(cell_size_1, return_sequences=True, stateful=False)(inputs)
    Batch_norm_1 = BatchNormalization()(Lstm_layer_1)
    Dropout_layer_1 = Dropout(d1)(Batch_norm_1)
    Lstm_layer_2 = LSTM(cell_size_2, return_sequences=False, stateful=False)(Dropout_layer_1)  # just halved
    Batch_norm_2 = BatchNormalization()(Lstm_layer_2)
    Drouput_layer_2 = Dropout(d2)(Batch_norm_2)
    predictions = Dense(n_classes, activation='softmax')(Drouput_layer_2)
    LSTM_base = Model(inputs=inputs, outputs=predictions)

    LSTM_base.summary()
    # optimizer
    optimizer = Adadelta(
    learning_rate=1.0,
    rho=0.8,
    epsilon=1e-7)      # Default , to prevent division by zero)

    LSTM_base.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Training the model
    history = LSTM_base.fit(x=X_train, y=Y_train,
                    validation_data=(X_valid, Y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False)
 
    fig, ax1 = plt.subplots()

    # Plot losses on the primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history.history['loss'], label='Train Loss', color='red', linestyle='-')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Combine legends from both axes
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)  # Legend outside the plot

    plt.title('Model Accuracy and Loss')
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    y_pred = LSTM_base.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    return Y_test, y_pred


# VAE Model:
"""Encoder-Decoder Architecture:
 - Input 3D
 - Encoder: 
    - LSTM layer 1 (return sequences), Batch_norm to handle covariate shifts, Dropout for regularization
    - LSTM layer 2 (returns single output), Batch_norm 2 (half cell size of 1)
 - Latent Space = VAE with KL-Divergence and Gaussian Distribution (variance and mean) - idea was for better generalisation, i.e. representation learning
 - Decoder:
    - Dense Layer with ReLu 
    - Skip Connection Batch_norm 2 and Dense Layer with ReLu (to ensure that temporal features still are kept) 
    - LSTM layer 3 (return sequences), Batch_norm 3, cell size = LSTM layer 1
    - LSTM layer 4 (returns single output), Batch_norm 4 (half cell size of 1), Dropout 2
    - Output Dense softmax for multiple up-down-sideways
 - Optimizer Adadelta
 - loss = sparse_categorical_crossentropy & KL divergence
 - metrics accuracy"""

def LSTM_autoencoder_rates_and_class(X_train,X_valid,X_test,Y_train,Y_valid,Y_test,epochs=10, 
                                     batch_size=64, d1=0.1, cell_size = 80, loss_weights=[1,0.1], latent_dim=10):
    
    tf.keras.backend.clear_session()
    n_classes = 3

    cell_size_1 = cell_size
    cell_size_2 = cell_size_1//2
    
    class KLDivergenceLayer(Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            self.add_loss(kl_loss)  # Add the KL loss to the layer's loss terms
            return kl_loss
        
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    Lstm_layer_1 = LSTM(cell_size_1,activation='tanh', return_sequences=True, stateful=False, use_cudnn=False)(inputs)
    Batch_norm_1 = BatchNormalization()(Lstm_layer_1)
    Dropout_layer_1 = Dropout(d1)(Batch_norm_1)
    Lstm_layer_2 = LSTM(cell_size_2,activation='tanh', return_sequences=False, stateful=False, use_cudnn=False)(Dropout_layer_1)  # just halved
    Batch_norm_2 = BatchNormalization()(Lstm_layer_2)
    
    # Latent space representation (mean and log variance)
    z_mean = Dense(latent_dim)(Batch_norm_2)
    z_log_var = Dense(latent_dim)(Batch_norm_2)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    decoder_hidden = Dense(cell_size_2, activation='relu')(z)
    reshaped_decoder_input = Reshape((1, cell_size_2))(decoder_hidden)
    Lstm_decoder_1 = LSTM(cell_size_1, return_sequences=True, use_cudnn=False, activation='tanh')(reshaped_decoder_input)
    Batch_norm_3 = BatchNormalization()(Lstm_decoder_1)
    Lstm_decoder_2 = LSTM(cell_size_2, return_sequences=False, use_cudnn=False, activation='tanh')(Batch_norm_3)
    Batch_norm_4 = BatchNormalization()(Lstm_decoder_2)
    Dropout_layer_2 = Dropout(d1)(Batch_norm_4)
    
    class_predictions = Dense(n_classes, activation='softmax',name="class")(Dropout_layer_2)

    kl_loss_layer = KLDivergenceLayer(name='kl_loss')([z_mean, z_log_var])

    LSTM_base_decoder = Model(inputs=inputs,outputs=[class_predictions, kl_loss_layer])
    # we will keep this as a standardized learning rate optimizer across all models
    optimizer = Adadelta(
    learning_rate=1.0,
    rho=0.8,
    epsilon=1e-7)
    
    
    LSTM_base_decoder.compile(
        optimizer=optimizer,
        loss=["sparse_categorical_crossentropy", lambda y_true, y_pred: tf.reduce_mean(y_pred)],
        loss_weights = loss_weights,
        metrics={'class': ['accuracy']})
    
    LSTM_base_decoder.summary()
    
    history = LSTM_base_decoder.fit(X_train, 
               [Y_train, np.zeros_like(Y_train)],
               validation_data=(X_valid, [Y_valid, np.zeros_like(Y_valid)]),
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=False)
    
    # Extract the specific keys for class_predictions and accuracy
    class_train_loss_key = 'class_loss'
    class_val_loss_key = 'val_class_loss'
    train_accuracy_key = 'class_accuracy'
    val_accuracy_key = 'val_class_accuracy'

    print(history.history.keys())

    # Create a plot with a secondary y-axis
    fig, ax1 = plt.subplots()

    # Plot the class losses on the primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history.history[class_train_loss_key], label='Train Loss', color='red', linestyle='-')
    ax1.plot(history.history[class_val_loss_key], label='Validation Loss', color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Add a secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history.history[train_accuracy_key], label='Train Accuracy', color='blue', linestyle='-')
    ax2.plot(history.history[val_accuracy_key], label='Validation Accuracy', color='blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Combine legends from both axes
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)  # Legend outside the plot

    plt.title('Class Loss and Accuracy Over Epochs')
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


    y_pred, y_placeholder = LSTM_base_decoder.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

# Attention Model:
"""Transformer based Architecture minus the pos encodings:
 - Input 3D
 - Multiattentionhead 1, Residual connection, Layer norm
 - Multiattentionhead 2, Residual connection, Layer norm
 - Dropout 
 - Linear feed forward
 - Dropout 2
 - Flatten
 - Linear feed forward 2
 - Output Dense softmax for multiple up-down-sideways
 - Optimizer Adadelta
 - loss = sparse_categorical_crossentropy
 - metrics accuracy"""

def Attention_model(X_train,X_valid,X_test,Y_train,Y_valid,Y_test,weights, epochs=100,
                     batch_size=100, d1=0.1, d2 = 0.05, cell_size = 80, attention_heads=4):

    tf.keras.backend.clear_session()
    n_classes = 3

    cell_size_1 = cell_size
    cell_size_2 = cell_size_1//2

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    attention_layer_1 = MultiHeadAttention(
    num_heads=attention_heads,
    key_dim=cell_size//attention_heads)(
    query=inputs,
    key=inputs,
    value=inputs)
    residual_1 = Add()([inputs, attention_layer_1])  # Add residual connection
    norm_1 = LayerNormalization()(residual_1)

    attention_layer_2 = MultiHeadAttention(
    num_heads=attention_heads,
    key_dim=cell_size//attention_heads)(
    query=norm_1,
    key=norm_1,
    value=norm_1)
    residual_2 = Add()([norm_1, attention_layer_2])
    norm_2 = LayerNormalization()(residual_2)
    dropout_1 = Dropout(d1)(norm_2)

    ffw = Dense(cell_size_2, activation="swish")(dropout_1)
    dropout_2 = Dropout(d2)(ffw)
    flatten_ = Flatten()(dropout_2)
    ffw_2 = Dense(cell_size_2,activation="swish")(flatten_)

    class_predictions = Dense(n_classes, activation='softmax',name="class")(ffw_2)
    Attention_base = Model(inputs=inputs, outputs=class_predictions)

    # we will keep this as a standardized learning rate optimizer across all models
    optimizer = Adadelta(
    learning_rate=1.0,
    rho=0.8,
    epsilon=1e-7)      # Default , to prevent division by zero)

    Attention_base.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    Attention_base.summary()

    history = Attention_base.fit(x=X_train, y=Y_train,
                    validation_data=(X_valid, Y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight = weights,
                    shuffle=False)


    fig, ax1 = plt.subplots()

    # Plot losses on the primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history.history['loss'], label='Train Loss', color='red', linestyle='-')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Combine legends from both axes
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)  # Legend outside the plot

    plt.title('Model Accuracy and Loss')
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    y_pred = Attention_base.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    return Y_test, y_pred

    

