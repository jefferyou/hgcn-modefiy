import tensorflow as tf
from utils import util
from sklearn.metrics import f1_score

class BaseHGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def my_training(loss, lr, l2_coef):
        all_update_ops = []
        # weight decay
        vars = tf.trainable_variables()  # all variables in training
        eucl_var = [tmp_vars for tmp_vars in vars if 'hyper' not in tmp_vars.name]
        l2_vars = [tmp_vars for tmp_vars in vars if 'bias' not in tmp_vars.name]
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]) * l2_coef
        # optimizer
        eucl_opt = tf.train.AdamOptimizer(learning_rate=lr)
        loss_sum = loss + lossL2
        # Euclidean
        # training op
        eucl_grads_vars = eucl_opt.compute_gradients(loss=loss_sum, var_list=eucl_var)
        eucl_clip_grads_vars = [(tf.clip_by_norm(grad, 1.), vars) for grad, vars in eucl_grads_vars]
        all_update_ops.append(eucl_opt.apply_gradients(grads_and_vars=eucl_clip_grads_vars))
        return tf.group(*all_update_ops)

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def my_masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        logits = tf.clip_by_value(logits, clip_value_min=1e-5, clip_value_max=1.)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    # def micro_f1(logits, labels, mask):
    #     """Accuracy with masking."""
    #     predicted = tf.round(tf.nn.sigmoid(logits))
    #
    #     # Use integers to avoid any nasty FP behaviour
    #     predicted = tf.cast(predicted, dtype=tf.int32)
    #     labels = tf.cast(labels, dtype=tf.int32)
    #     mask = tf.cast(mask, dtype=tf.int32)
    #
    #     # expand the mask so that broadcasting works ([nb_nodes, 1])
    #     mask = tf.expand_dims(mask, -1)
    #
    #     # Count true positives, true negatives, false positives and false negatives.
    #     tp = tf.count_nonzero(predicted * labels * mask)
    #     tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
    #     fp = tf.count_nonzero(predicted * (labels - 1) * mask)
    #     fn = tf.count_nonzero((predicted - 1) * labels * mask)
    #
    #     # Calculate accuracy, precision, recall and F1 score.
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     fmeasure = (2 * precision * recall) / (precision + recall)
    #     fmeasure = tf.cast(fmeasure, tf.float32)
    #     return fmeasure

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        # Apply mask to isolate the relevant nodes
        mask_bool = tf.cast(mask, dtype=tf.bool)

        # Get masked logits and labels
        masked_logits = tf.boolean_mask(logits, mask_bool)
        masked_labels = tf.boolean_mask(labels, mask_bool)

        # Perform argmax for multi-class classification
        pred_class = tf.argmax(masked_logits, axis=1)
        true_class = tf.argmax(masked_labels, axis=1)

        # Convert to same type for comparison
        pred_class = tf.cast(pred_class, tf.int32)
        true_class = tf.cast(true_class, tf.int32)

        # Calculate true positives (correct predictions)
        correct_predictions = tf.equal(pred_class, true_class)
        tp = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))

        # Calculate F1 score (for multi-class, micro F1 is equivalent to accuracy)
        f1 = tp / tf.cast(tf.shape(masked_labels)[0], tf.float32)

        return tf.cast(f1, tf.float32)


    def standard_accuracy(logits, labels, mask):
        """Compute standard accuracy with masking."""
        # Get predicted and true labels
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)

        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)

    def weighted_f1(logits, labels, mask):
        """Compute weighted F1 score."""
        # This is a placeholder. In practice, it's better to use sklearn's f1_score with average='weighted'
        # as we did in the main function, since implementing this properly in TensorFlow is complex
        predicted = tf.argmax(logits, 1)
        true_labels = tf.argmax(labels, 1)

        # Apply mask - get only relevant predictions and labels
        mask = tf.cast(mask, dtype=tf.bool)
        predicted = tf.boolean_mask(predicted, mask)
        true_labels = tf.boolean_mask(true_labels, mask)

        # Note: The proper implementation would require computing precision and recall per class,
        # weighting by class support, and then calculating the F1. For simplicity, we're using
        # sklearn's implementation in the main loop.

        return tf.constant(0.0)  # Placeholder




