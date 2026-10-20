import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential


# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def call(self, inputs, training=False):

        attention_output = self.attention(
            inputs,
            inputs
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.norm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.norm2(
            out1 + ffn_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })

        return config


# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

with open("models/tokenizerbest.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# --------------------------------------------------
# Load Intent Mapping
# --------------------------------------------------

with open("models/id2label.pkl", "rb") as file:
    id2label = pickle.load(file)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = tf.keras.models.load_model(
    "models/best_model.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock
    },
    compile=False
)


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_intent(text):

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=50,
        padding="post"
    )

    # Prediction
    prediction = model.predict(
        padded,
        verbose=0
    )[0]

    # Highest probability
    predicted_id = int(np.argmax(prediction))

    confidence = float(prediction[predicted_id])

    # Intent name
    intent = id2label[predicted_id]

    return {
        "intent_id": predicted_id,
        "intent": intent,
        "confidence": confidence
    }
