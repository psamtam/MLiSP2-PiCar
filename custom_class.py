import tensorflow as tf
import keras as k
from tensorflow.keras import backend as K


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives_0 = self.add_weight(name="tp_0", initializer="zeros")
        self.true_positives_1 = self.add_weight(name="tp_1", initializer="zeros")
        self.total_0 = self.add_weight(name="total_0", initializer="zeros")
        self.total_1 = self.add_weight(name="total_1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.cast(tf.round(y_pred), tf.int32)
        y_pred = tf.squeeze(y_pred)

        mask_0 = tf.equal(y_true, 0)
        true_pos_0 = tf.reduce_sum(
            tf.cast(tf.logical_and(mask_0, tf.equal(y_pred, 0)), tf.float32)
        )
        total_0 = tf.reduce_sum(tf.cast(mask_0, tf.float32))

        mask_1 = tf.equal(y_true, 1)
        true_pos_1 = tf.reduce_sum(
            tf.cast(tf.logical_and(mask_1, tf.equal(y_pred, 1)), tf.float32)
        )
        total_1 = tf.reduce_sum(tf.cast(mask_1, tf.float32))

        self.true_positives_0.assign_add(true_pos_0)
        self.true_positives_1.assign_add(true_pos_1)
        self.total_0.assign_add(total_0)
        self.total_1.assign_add(total_1)

    def result(self):
        recall_0 = self.true_positives_0 / (self.total_0 + tf.keras.backend.epsilon())
        recall_1 = self.true_positives_1 / (self.total_1 + tf.keras.backend.epsilon())
        return (recall_0 + recall_1) / 2.0

    def reset_state(self):
        self.true_positives_0.assign(0.0)
        self.true_positives_1.assign(0.0)
        self.total_0.assign(0.0)
        self.total_1.assign(0.0)


@k.saving.register_keras_serializable()
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight_dict, reduction="sum_over_batch_size", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.class_weight_dict = class_weight_dict
        self.class_weight = tf.convert_to_tensor(
            [self.class_weight_dict[i] for i in sorted(self.class_weight_dict.keys())],
            dtype=tf.float32,
        )

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Clip 0s and 1s

        bce = tf.keras.losses.BinaryCrossentropy()
        bce_loss = bce(y_true, y_pred)

        weights = tf.gather(self.class_weight, tf.cast(y_true, tf.int32))

        return tf.reduce_mean(weights * bce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"class_weight_dict": self.class_weight_dict})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, class_weight_dict, reduction="sum_over_batch_size", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.class_weight_dict = class_weight_dict
        self.class_weight = tf.convert_to_tensor(
            [self.class_weight_dict[i] for i in sorted(self.class_weight_dict.keys())],
            dtype=tf.float32,
        )

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Clip 0s and 1s

        # Calculate cross-entropy loss directly with one-hot encoded inputs
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Get weights based on true classes from one-hot encoding
        class_indices = tf.argmax(y_true, axis=-1)
        weights = tf.gather(self.class_weight, class_indices)

        return tf.reduce_mean(weights * ce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"class_weight_dict": self.class_weight_dict})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@k.saving.register_keras_serializable()
class WeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, angle_weight_dict, reduction="sum_over_batch_size", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.angle_weight_dict = angle_weight_dict
        self.weights = tf.constant(
            list(self.angle_weight_dict.values()), dtype=tf.float32
        )
        self.keys = tf.constant(list(self.angle_weight_dict.keys()), dtype=tf.float32)

    def call(self, y_true, y_pred):
        squared_error = tf.square(y_true - y_pred)

        def get_weight(y):
            diffs = tf.abs(y - self.keys)
            idx = tf.argmin(diffs)
            return tf.gather(self.weights, idx)

        batch_weights = tf.map_fn(get_weight, y_true)

        weighted_squared_diff = squared_error * batch_weights

        return tf.reduce_mean(weighted_squared_diff)

    def get_config(self):
        config = super().get_config()
        config.update({"angle_weight_dict": self.angle_weight_dict})
        return config

    @classmethod
    def from_config(cls, config):
        angle_weight_dict = {float(k): v for k, v in config["angle_weight_dict"].items()}
        config["angle_weight_dict"] = angle_weight_dict
        return cls(**config)

class BalancedAccuracyMetrics:
    """Class containing balanced accuracy metric implementations"""
    
    @staticmethod
    @tf.function
    def balanced_accuracy(y_true, y_pred):
        """
        Calculate balanced accuracy for multi-class classification
        
        Args:
            y_true: Ground truth labels (tensor)
            y_pred: Predicted probabilities or logits (tensor)
        
        Returns:
            Balanced accuracy score
        """
        # Ensure inputs are in the correct format
        if len(y_true.shape) > 1:
            y_true_labels = tf.argmax(y_true, axis=-1)
        else:
            y_true_labels = tf.cast(y_true, tf.int64)
            
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        
        # Cast to float32 for calculations
        y_true = tf.cast(y_true_labels, tf.float32)
        y_pred = tf.cast(y_pred_labels, tf.float32)
        
        # Get unique classes
        classes = tf.unique(y_true)[0]
        num_classes = tf.cast(tf.shape(classes)[0], tf.float32)
        
        # Vectorized computation of per-class accuracy
        def compute_per_class_acc():
            def class_acc_fn(c):
                class_mask = tf.equal(y_true, tf.cast(c, y_true.dtype))
                correct = tf.reduce_sum(tf.cast(
                    tf.boolean_mask(tf.equal(y_pred, y_true), class_mask),
                    tf.float32))
                total = tf.reduce_sum(tf.cast(class_mask, tf.float32))
                return tf.cond(
                    total > 0,
                    lambda: correct / total,
                    lambda: 0.0
                )
            
            return tf.map_fn(
                class_acc_fn,
                tf.range(tf.cast(num_classes, tf.int32)),
                fn_output_signature=tf.float32
            )
        
        per_class_acc = compute_per_class_acc()
        return tf.reduce_mean(per_class_acc)

    @staticmethod
    def binary_balanced_accuracy(y_true, y_pred):
        """
        Calculate balanced accuracy for binary classification
        
        Args:
            y_true: Ground truth labels (tensor)
            y_pred: Predicted probabilities (tensor)
        
        Returns:
            Balanced accuracy score
        """
        # Convert probabilities to binary predictions
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # True positives and true negatives
        tp = tf.reduce_sum(y_true * y_pred_binary)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
        
        # Positive and negative counts
        p = tf.reduce_sum(y_true)
        n = tf.reduce_sum(1 - y_true)
        
        # Sensitivity and specificity
        sensitivity = tp / (p + K.epsilon())
        specificity = tn / (n + K.epsilon())
        
        # Balanced accuracy
        return (sensitivity + specificity) / 2
