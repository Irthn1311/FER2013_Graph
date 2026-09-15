"""Two-device graph data parallelism with a single authoritative optimizer.

Each device owns a model copy. Forward/backward branches are placed explicitly
in one compiled TF graph. Their globally normalized gradients are summed on
GPU:0 before the unchanged optimizer clips/updates once. The updated weights
are copied to GPU:1 before the step returns. This avoids relying on private
distributed hooks of the frozen custom AdamW or clipping per replica.
"""

import tensorflow as tf

from pixel_gnn_only.batching import GraphBatchGenerator
from pixel_gnn_only.execution import validate_gradient_contract
from pixel_gnn_only.model import PixelGNNOnly


def variable_devices(model):
    return sorted({variable.handle.device for variable in model.trainable_variables})


def shard_graph_batch(batch, graph_start, graph_end):
    """Slice complete, contiguous graphs and rebase edge/node graph indices."""
    with tf.device("/CPU:0"):
        node_start = tf.reduce_sum(batch["graph_node_counts"][:graph_start])
        node_end = tf.reduce_sum(batch["graph_node_counts"][:graph_end])
        edge_start = tf.reduce_sum(batch["graph_edge_counts"][:graph_start])
        edge_end = tf.reduce_sum(batch["graph_edge_counts"][:graph_end])
        return {
            "node_features": batch["node_features"][node_start:node_end],
            "edge_index": batch["edge_index"][:, edge_start:edge_end] - node_start,
            "edge_features": batch["edge_features"][edge_start:edge_end],
            "node_graph_index": batch["node_graph_index"][node_start:node_end] - tf.cast(graph_start, tf.int64),
            "graph_node_counts": batch["graph_node_counts"][graph_start:graph_end],
            "graph_edge_counts": batch["graph_edge_counts"][graph_start:graph_end],
            "labels": batch["labels"][graph_start:graph_end],
            "sample_ids": batch["sample_ids"][graph_start:graph_end],
            "image_48": batch["image_48"][graph_start:graph_end],
        }


def sync_replica(primary, replica):
    if len(primary.trainable_variables) != len(replica.trainable_variables):
        raise RuntimeError("Replica variable-count mismatch")
    with tf.device("/GPU:1"):
        for source, target in zip(primary.trainable_variables, replica.trainable_variables):
            if source.shape != target.shape:
                raise RuntimeError("Replica variable-shape mismatch")
            target.assign(tf.cast(source, tf.float32))
        # Reads depend on all assignments in tf.function's automatic control
        # dependencies; the returned token makes synchronization part of the step.
        return tf.group(*[tf.identity(variable) for variable in replica.trainable_variables])


def create_replica(primary, example_batch, gpu_count):
    if gpu_count == 1:
        return None
    with tf.device("/GPU:1"):
        replica = PixelGNNOnly()
        replica(shard_graph_batch(example_batch, 0, 1), training=False)
    sync_replica(primary, replica)
    if any("GPU:0" not in device for device in variable_devices(primary)):
        raise RuntimeError("Primary model variables were not placed on GPU:0")
    if any("GPU:1" not in device for device in variable_devices(replica)):
        raise RuntimeError("Replica model variables were not placed on GPU:1")
    count = int(example_batch["labels"].shape[0])
    print(f"[PARALLEL] GPU:0 + GPU:1; example batch {count} = {(count + 1) // 2} + {count // 2}; weight replica ready", flush=True)
    return replica


def replica_gradients(model, batch, optimizer, total_examples, training=True):
    with tf.GradientTape() as tape:
        logits = tf.cast(model(batch, training=training)["logits"], tf.float32)
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch["labels"], logits, from_logits=True)
        # Use the actual global batch count, including the unchanged short last batch.
        loss = tf.reduce_sum(losses) / tf.cast(total_examples, tf.float32)
        if hasattr(optimizer, "scale_loss"):
            scaled = optimizer.scale_loss(loss)
        elif hasattr(optimizer, "get_scaled_loss"):
            scaled = optimizer.get_scaled_loss(loss)
        else:
            scaled = loss
    gradients = tape.gradient(scaled, model.trainable_variables)
    validate_gradient_contract(gradients, model.trainable_variables)
    return loss, gradients


def build_parallel_train_step(primary, replica, optimizer, training=True):
    if replica is None:
        from pixel_gnn_only.execution import build_restricted_graph_train_step
        return build_restricted_graph_train_step(primary, optimizer, GraphBatchGenerator.output_signature(), training=training)

    @tf.function(autograph=False, jit_compile=False,
                 input_signature=[GraphBatchGenerator.output_signature()], reduce_retracing=True)
    def train_step(batch):
        with tf.device("/CPU:0"):
            count = tf.shape(batch["labels"])[0]
            tf.debugging.assert_positive(count)

        def single_graph():
            with tf.device("/GPU:0"):
                return replica_gradients(primary, batch, optimizer, count, training=training)

        def two_devices():
            with tf.device("/CPU:0"):
                midpoint = (count + 1) // 2
                left = shard_graph_batch(batch, 0, midpoint)
                right = shard_graph_batch(batch, midpoint, count)
            with tf.device("/GPU:0"):
                loss_left, gradients_left = replica_gradients(primary, left, optimizer, count, training=training)
            with tf.device("/GPU:1"):
                loss_right, gradients_right = replica_gradients(replica, right, optimizer, count, training=training)
            with tf.device("/GPU:0"):
                return loss_left + loss_right, [a + b for a, b in zip(gradients_left, gradients_right)]

        loss, gradients = tf.cond(count > 1, two_devices, single_graph)
        with tf.device("/GPU:0"):
            if not hasattr(optimizer, "scale_loss") and hasattr(optimizer, "get_unscaled_gradients"):
                gradients = optimizer.get_unscaled_gradients(gradients)
            validate_gradient_contract(gradients, primary.trainable_variables)
            optimizer.apply_gradients(zip(gradients, primary.trainable_variables))
        synchronized = sync_replica(primary, replica)
        with tf.control_dependencies([synchronized]):
            return tf.identity(loss)

    return train_step


def build_parallel_evaluation_step(primary, replica):
    if replica is None:
        from pixel_gnn_only.evaluator import build_compiled_evaluation_step
        return build_compiled_evaluation_step(primary)

    def predict(model, batch):
        output = model(batch, training=False)
        logits = tf.cast(output["logits"], tf.float32)
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch["labels"], logits, from_logits=True)
        return tf.reduce_sum(losses), output["probabilities"]

    @tf.function(autograph=False, jit_compile=False,
                 input_signature=[GraphBatchGenerator.output_signature()], reduce_retracing=True)
    def evaluate_step(batch):
        count = tf.shape(batch["labels"])[0]

        def single_graph():
            with tf.device("/GPU:0"):
                total, probabilities = predict(primary, batch)
            return total / tf.cast(count, tf.float32), probabilities

        def two_devices():
            midpoint = (count + 1) // 2
            left = shard_graph_batch(batch, 0, midpoint)
            right = shard_graph_batch(batch, midpoint, count)
            with tf.device("/GPU:0"):
                loss_left, probabilities_left = predict(primary, left)
            with tf.device("/GPU:1"):
                loss_right, probabilities_right = predict(replica, right)
            with tf.device("/CPU:0"):
                return ((loss_left + loss_right) / tf.cast(count, tf.float32),
                        tf.concat([probabilities_left, probabilities_right], axis=0))

        return tf.cond(count > 1, two_devices, single_graph)

    return evaluate_step
