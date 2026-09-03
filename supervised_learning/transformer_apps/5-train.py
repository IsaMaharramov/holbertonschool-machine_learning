#!/usr/bin/env python3
"""
Custom loop logic to train a Transformer Model
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom Learning Rate Schedule class"""
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Custom Sparse Categorical Crossentropy applying padding masks"""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation

    Args:
        N: number of blocks in the encoder and decoder
        dm: dimensionality of the model
        h: number of heads
        hidden: number of hidden units in the fully connected layers
        max_len: maximum number of tokens per sequence
        batch_size: batch size for training
        epochs: number of epochs to train for

    Returns:
        The trained model
    """
    # 1. Initialize Pipeline & Vocabulary Variables
    data = Dataset(batch_size, max_len)
    
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2
    
    # 2. Build Transformer Instance
    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len)
    
    # 3. Optimizers & Callbacks Configuration
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    # 4. Train Step (compiles down to TF graph for speed)
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = \
            create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True, enc_padding_mask,
                                      combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)
            
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        
        # Accuracy doesn't matter padding wise since target contains 0s
        # Which throws metrics for padded values away implicitly via logic matching
        train_accuracy(tar_real, predictions)

    # 5. Core Epoch Training Loop
    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        for batch, (inp, tar) in enumerate(data.data_train):
            train_step(inp, tar)
            
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {batch}: loss "
                      f"{train_loss.result()} accuracy "
                      f"{train_accuracy.result()}")
                
        print(f"Epoch {epoch + 1}: loss {train_loss.result()} "
              f"accuracy {train_accuracy.result()}")
        
    return transformer
