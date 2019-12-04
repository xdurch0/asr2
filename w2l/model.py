import os
import time
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import numpy as np


GRIDS = {16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8), 256: (16, 16),
         512: (32, 16), 1024: (32, 32), 2048: (64, 32)}


class W2L:
    def __init__(self, model_dir, vocab_size, n_channels, data_format,
                 reg=(None, 0.)):
        if data_format not in ["channels_first", "channels_last"]:
            raise ValueError("Invalid data type specified: {}. Use either "
                             "channels_first or channels_last.".format(data_format))

        self.model_dir = model_dir
        self.data_format = data_format
        self.cf = self.data_format == "channels_first"
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.regularizer_type = reg[0]
        self.regularizer_coeff = reg[1]

        if os.path.isdir(model_dir) and os.listdir(model_dir):
            print("Model directory already exists. Loading last model...")
            last = self.get_last_model(model_dir)
            self.model = tf.keras.models.load_model(
                os.path.join(model_dir, last))
            print("...loaded {}.".format(last))
        else:
            print("Model directory does not exist. Creating new model...")
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            self.model = self.make_w2l_model()

    def make_w2l_model(self):
        """Creates a Keras model that does the W2L forward computation.

        Just goes from mel spectrogram input to logits output.

        Parameters:
            vocab_size: Size of the vocabulary. Output layer will have this size + 1
                        for the blank label.
            reg: Regularizer type to use, or None for no regularizer.


        Returns:
            Keras sequential model.

        TODO could allow model configs etc. For now, architecture is hardcoded

        """
        channel_ax = 1 if self.cf else -1

        if self.regularizer_type:
            reg_target, reg_type, reg_edges, reg_size = self.regularizer_type.split("_")
            reg_fn_builder = lambda n_f: sebastians_magic_trick(
                diff_norm=reg_type, weight_norm="l2", grid_dims=GRIDS[n_f],
                neighbor_size=int(reg_size),
                cf=(self.cf and reg_target == "act"),
                edges=reg_edges)
        else:
            reg_target = None

        def reg_conv1d(n_f, w_f, stride):
            return layers.Conv1D(
                n_f, w_f, stride, padding="same", data_format=self.data_format,
                use_bias=False,
                kernel_regularizer=reg_fn_builder(
                    n_f) if reg_target == "weight" else None,
                activity_regularizer=reg_fn_builder(
                    n_f) if reg_target == "act" else None)

        layer_list = [
            reg_conv1d(256, 48, 2),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(256, 7, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(2048, 32, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            reg_conv1d(2048, 1, 1),
            layers.BatchNormalization(channel_ax),
            layers.ReLU(),
            layers.Conv1D(self.vocab_size + 1, 1, 1, "same", self.data_format)
            ]

        # w2l = tf.keras.Sequential(layer_list, name="w2l")

        inp = tf.keras.Input((self.n_channels, None) if self.cf
                             else (None, self.n_channels))
        layer_outputs = [inp]
        for layer in layer_list:
            layer_outputs.append(layer(layer_outputs[-1]))
        # only include relu layers in outputs
        relevant = layer_outputs[3::3] + [layer_outputs[-1]]

        w2l = tf.keras.Model(inputs=inp, outputs=relevant)

        return w2l

    def forward(self, audio, training=False, return_all=False):
        """Simple forward pass of a W2L model to compute logits.

        Parameters:
            audio: Tensor of mel spectrograms, channels_first!
            training: Bool, if true assuming training mode otherwise inference.
                      Important for batchnorm to work properly.
            return_all: Bool, if true, return list of all layer activations
                        (post-relu), with the logits at the very end.

        Returns:
            Result of applying model to audio (list or tensor depending on
            return_all).

        """
        if not self.cf:
            audio = tf.transpose(audio, [0, 2, 1])

        out = self.model(audio, training=training)
        if return_all:
            return out
        else:
            return out[-1]

    def train_step(self, audio, audio_length, transcrs, transcr_length,
                   optimizer, on_gpu):
        """Implements train step of the W2L model.

        Parameters:
            audio: Tensor of mel spectrograms, channels_first!
            audio_length: "True" length of each audio clip.
            transcrs: Tensor of transcriptions (indices).
            transcr_length: "True" length of each transcription.
            optimizer: Optimizer instance to do training with.
            on_gpu: Bool, whether running on GPU. This changes how the
                    transcriptions are handled. Currently ignored!!

        Returns:
            Loss value.

        """
        with tf.GradientTape() as tape:
            logits = self.forward(audio, training=True, return_all=False)
            # after this we need logits in shape time x batch_size x vocab_size
            if self.cf:  # bs x v x t -> t x bs x v
                logits_tm = tf.transpose(logits, [2, 0, 1],
                                         name="logits_time_major")
            else:  # channels last: bs x t x v -> t x bs x v
                logits_tm = tf.transpose(logits, [1, 0, 2],
                                         name="logits_time_major")

            audio_length = tf.cast(audio_length / 2, tf.int32)

            if False:  #on_gpu:  # this seems to be slow so we don't use it
                ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=transcrs, logits=logits_tm, label_length=transcr_length,
                    logit_length=audio_length, logits_time_major=True,
                    blank_index=0), name="avg_loss")
            else:
                transcrs_sparse = dense_to_sparse(transcrs, sparse_val=-1)
                ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=transcrs_sparse, logits=logits_tm, label_length=None,
                    logit_length=audio_length, logits_time_major=True,
                    blank_index=0), name="avg_loss")

            if self.regularizer_coeff:
                avg_reg_loss = tf.math.add_n(self.model.losses) / len(self.model.losses)
                loss = ctc_loss + self.regularizer_coeff * avg_reg_loss
            else:
                loss = ctc_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        #self.annealer.update_history(loss)

        return loss

    def train_full(self, dataset, steps, adam_params, on_gpu):
        """Full training logic for W2L.

        Parameters:
            dataset: tf.data.Dataset as produced in input.py.
            steps: Number of training steps.
            adam_params: List/tuple of four parameters for Adam: learning rate,
                         beta1, beta2, epsilon.
            on_gpu: Bool, whether running on a GPU.

        """
        step = 0
        # TODO more flexible checkpointing. this will simply do 10 checkpoints overall
        check_freq = steps // 10
        data_step_limited = dataset.take(steps)

        #self.annealer = AnnealIfStuck(adam_params[0], 0.1, 20000)
        opt = tf.optimizers.Adam(*adam_params)

        audio_shape = [None, self.n_channels, None] if self.cf \
            else [None, None, self.n_channels]
        train_fn = lambda w, x, y, z: self.train_step(
            w, x, y, z, opt, on_gpu)
        graph_train = tf.function(
            train_fn, input_signature=[tf.TensorSpec(audio_shape, tf.float32),
                                       tf.TensorSpec([None], tf.int32),
                                       tf.TensorSpec([None, None], tf.int32),
                                       tf.TensorSpec([None], tf.int32)])
        # graph_train = train_fn  # skip tf.function

        start = time.time()
        for features, labels in data_step_limited:
            ctc = graph_train(features["audio"], features["length"],
                              labels["transcription"], labels["length"])
            if not step % 500:
                stop = time.time()
                print("Step: {}. CTC: {}".format(step, ctc.numpy()))
                print("{} seconds passed...".format(stop-start))
            if not step % check_freq:
                print("Saving checkpoint...")
                self.model.save(os.path.join(self.model_dir, str(step).zfill(6) + ".h5"))
            step += 1

        self.model.save(os.path.join(self.model_dir, "final.h5"))

    def decode(self, audio, audio_length, return_intermediate=False):
        """Wrapper to decode using W2L model.

        Parameters:
            audio: Tensor of mel spectrograms, channels_first!
            audio_length: "True" length of each audio clip.
            return_intermediate: Bool; if true, return intermediate layer results
                                 in addition to the decodings.

        Returns:
            Sparse or dense tensor with the top predictions.
            If return_intermediate is True, output is a tuple, first element being
            the predictions and second element a list of intermediate outputs.

        """
        forward = self.forward(audio, training=False,
                               return_all=return_intermediate)
        if return_intermediate:
            logits = forward[-1]
        else:
            logits = forward

        if self.cf:
            logits = tf.transpose(logits, [2, 0, 1])
        else:
            logits = tf.transpose(logits, [1, 0, 2])

        decoded = self.ctc_decode_top(logits, audio_length, pad_val=-1)
        if return_intermediate:
            return decoded, forward
        else:
            return decoded

    def ctc_decode_top(self, logits, seq_lengths, beam_width=100, pad_val=-1,
                       as_sparse=False):
        """Simpler version of ctc decoder that only returns the top result.

        Parameters:
            logits: Passed straight to ctc decoder. This has to be time-major and
                    channels_last!!
            seq_lengths: Same.
            beam_width: Same.
            pad_val: Value to use to pad dense tensor. No effect if as_sparse is
                     True.
            as_sparse: If True, return results as sparse tensor.

        Returns:
            Sparse or dense tensor with the top predictions.

        """
        with tf.name_scope("decoding"):
            decoded_sparse_list, _ = tf.nn.ctc_beam_search_decoder(
                logits, seq_lengths//2, beam_width=beam_width, top_paths=1)
            decoded_sparse = decoded_sparse_list[0]
            decoded_sparse = tf.cast(decoded_sparse, tf.int32)
            if as_sparse:
                return decoded_sparse
            else:
                # this should result in a bs x t matrix of predicted classes
                return tf.sparse.to_dense(decoded_sparse,
                                          default_value=pad_val,
                                          name="dense_decoding")

    def get_last_model(self, model_dir):
        ckpts = [file for file in os.listdir(model_dir) if file.endswith(".h5")]
        if "final.h5" in ckpts:
            return "final.h5"
        else:
            return sorted(ckpts)[-1]


class AnnealIfStuck(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, factor, n_steps):
        """Anneal the learning rate if loss doesn't decrease anymore.

        Refer to
        http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html.

        Parameters:
            base_lr: LR to start with.
            factor: By what to multiply in case we're stuck.
            n_steps: How often to check if we're stuck.

        """
        super(AnnealIfStuck, self).__init__()
        self.n_steps = n_steps
        self.lr = base_lr
        self.factor = factor
        self.loss_history = tf.Variable(
            np.zeros(n_steps), trainable=False, dtype=tf.float32,
            name="loss_history")

    def __call__(self, step):
        if tf.logical_or(tf.greater(tf.mod(step, self.n_steps), 0),
                         tf.equal(step, 0)):
            pass
        else:
            x1 = tf.range(self.n_steps, dtype=tf.float32, name="x")
            x2 = tf.ones([self.n_steps], dtype=tf.float32, name="bias")
            x = tf.stack((x1, x2), axis=1, name="input")
            slope_bias = tf.linalg.lstsq(x, self.loss_history[:, tf.newaxis],
                                         name="solution")
            slope = slope_bias[0][0]
            bias = slope_bias[1][0]
            preds = slope * x1 + bias

            data_var = 1 / (self.n_steps - 2) * tf.reduce_sum(tf.square(self.loss_history -
                                                             preds))
            dist_var = 12 * data_var / (self.n_steps ** 3 - self.n_steps)
            dist = tfp.distributions.Normal(slope, tf.sqrt(dist_var),
                                            name="slope_distribution")
            prob_decreasing = dist.cdf(0., name="prob_below_zero")

            if tf.less_equal(prob_decreasing, 0.5):
                self.lr *= self.factor
        return self.lr

    def update_history(self, new_val):
        self.loss_history.assign(tf.concat((self.loss_history[1:], [new_val]),
                                           axis=0))


def dense_to_sparse(dense_tensor, sparse_val=-1):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.

    Returns:
        SparseTensor equivalent to the dense input.

    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def sebastians_magic_trick(diff_norm, weight_norm, grid_dims, neighbor_size,
                           cf, edges):
    """Creates a neighborhood distance regularizer.

    Parameters:
        diff_norm: How to compute differences/distances between filters.
                   Can be "l1", "l2" or "linf" for respective norms, or "cos"
                   for cosine distance..
        weight_norm: How to compute neighborhood weightings, i.e. how points
                     further away in the neighborhood play into the overall
                     penalty. Options same as for diff_norm, except for "cos".
        grid_dims: 2-tuple or list giving the desired grid dimensions. Has to
                   match the number of filters for the layer to regularize.
        neighbor_size: int, giving the size of the neighborhood. Must be odd.
                       E.g. giving 3 here will cause each filter to treat the
                       immediately surrounding filters (including diagonally)
                       as its neighborhood.
        cf: Whether the regularizer target will be channels_first. Should only
            be true if we are regularizing activation maps, not filter weights,
            and channels_first data format is used.
        edges: String, how to treat edges. See CLI file for options.

    """
    if not neighbor_size % 2:
        raise ValueError("Neighborhood is not odd; this would mean no middle "
                         "point!")
    if edges not in {"no", "occ", "wrap", "mirror"}:
        raise ValueError("Invalid edge option specified: {}. Valid are 'no', "
                         "'occ' and 'wrap'.".format(edges))

    # first we compute the possible offsets around a given point
    neighbors_per_direction = (neighbor_size - 1) // 2
    neighbor_offsets = []
    for offset_x in range(-neighbors_per_direction,
                          neighbors_per_direction + 1):
        for offset_y in range(-neighbors_per_direction,
                              neighbors_per_direction + 1):
            if offset_x == 0 and offset_y == 0:
                continue  # skip center
            neighbor_offsets.append([offset_x, offset_y])
    neighbor_offsets = np.asarray(neighbor_offsets, dtype=np.int32)

    len_x = grid_dims[0]
    len_y = grid_dims[1]
    filters_total = len_x * len_y

    # get neighbors for each filter
    neighbor_lists = []
    for ci in range(filters_total):
        neighbors = []
        # derive x and y coordinate in filter space
        cy = ci // len_x
        cx = ci % len_x
        for offset in neighbor_offsets:
            offset_x = cx + offset[0]
            offset_y = cy + offset[1]

            if edges == "wrap":
                if offset_x < 0:
                    offset_x += len_x
                elif offset_x >= len_x:
                    offset_x -= len_x
                if offset_y < 0:
                    offset_y += len_y
                elif offset_y >= len_y:
                    offset_y -= len_y

            elif edges == "mirror":
                if offset_x < 0:
                    offset_x = -offset_x
                elif offset_x >= len_x:
                    d = offset_x - (len_x - 1)
                    offset_x = len_x - 1 - d

                if offset_y < 0:
                    offset_y -= offset_y
                elif offset_y >= len_y:
                    d = offset_y - (len_y - 1)
                    offset_y = len_y - 1 - d

            if 0 <= offset_x < len_x and 0 <= offset_y < len_y:
                # add neighbor if valid coordinate
                ni = offset_y * len_x + offset_x
                neighbors.append(ni)
        neighbor_lists.append(neighbors)

    # filter neighbor lists to only contain full neighborhoods
    center_ids = []
    neighbor_ids = []
    for ci, nis in enumerate(neighbor_lists):
        # e.g. in a 5x5 grid there are max. 24 neighbors
        if len(nis) == neighbor_size**2 - 1:
            center_ids.append(ci)
            neighbor_ids.append(nis)
    center_ids = np.asarray(center_ids, dtype=np.int32)
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int32)

    # weigh points further away in the neighborhood less
    neighbor_weights = []
    for offsets in neighbor_offsets:
        if weight_norm == "l1":
            d = np.abs(offsets).sum()
        elif weight_norm == "l2":
            d = np.sqrt((offsets*offsets).sum())
        elif weight_norm == "linf":
            d = np.abs(offsets).max()
        else:
            raise ValueError("Invalid weight norm specified: {}. "
                             "Valid are 'l1', 'l2', "
                             "'linf'.".format(weight_norm))
        w = 1. / d
        neighbor_weights.append(w)
    neighbor_weights = np.asarray(neighbor_weights, dtype=np.float32)

    if edges == "occ":
        # less often occurring positions are weighted more
        index_occurrences = defaultdict(int)
        for neighborhood in neighbor_ids:
            for neighbor in neighborhood:
                index_occurrences[neighbor] += 1

        max_occur = max(index_occurrences.values())

        neighbor_weights_occ = np.zeros(
            (len(neighbor_ids), len(neighbor_weights)),
            dtype=np.float32)
        for row in range(len(neighbor_ids)):
            for col in range(len(neighbor_weights)):
                occs_here = index_occurrences[neighbor_ids[row][col]]
                neighbor_weights_occ[row, col] = (neighbor_weights[col] *
                                                  max_occur / occs_here)
        neighbor_weights = neighbor_weights_occ
    else:
        neighbor_weights = np.tile(neighbor_weights, reps=[len(center_ids), 1])

    neighbor_weights /= neighbor_weights.sum()  # normalize to sum=1
    # neighbor_weights /= np.sqrt((neighbor_weights ** 2).sum())  # normalize to length=1

    # now convert numpy arrays to tf constants
    with tf.name_scope("nd_regularizer_sebastian"):
        tf_neighbor_weights = tf.constant(neighbor_weights,
                                          name='neighbor_weights')
        tf_center_ids = tf.constant(center_ids, name='center_ids')
        tf_neighbor_ids = tf.constant(neighbor_ids, name='neighbor_ids')

        def neighbor_distance(inputs):
            """If cf is true we assume channels first. Otherwise last, this also
            covers the case where the inputs are filter weights!
            """
            if not cf:
                n_filters = inputs.shape.as_list()[-1]
            else:
                n_filters = inputs.shape.as_list()[1]
            if n_filters != filters_total:
                raise ValueError(
                    "Unsuitable grid for weight {}. "
                    "Grid dimensions: {}, {} for a total of {} entries. "
                    "Filters in weight: {}.".format(
                        inputs.name, len_x, len_y, filters_total, n_filters))
            # reshape to n_filters x d
            if not cf:
                inputs = tf.reshape(inputs, [-1, n_filters])
                inputs = tf.transpose(inputs)
            else:
                perm = [1, 0] + list(range(2, len(inputs.shape)))
                inputs = tf.transpose(inputs, perm)
                inputs = tf.reshape(inputs, [n_filters, -1])

            if diff_norm == "l1":
                # to prevent weights from just shrinking (instead of getting
                # more similar) we apply a "global" normalization
                # note that local normalization (normalizing each filter
                # separately) would ignore scale differences between filters,
                # thus not forcing them to be "equal" properly
                inputs = inputs / tf.norm(inputs, ord=1)
            elif diff_norm == "l2":
                inputs = inputs / tf.norm(inputs)
            elif diff_norm == "linf":
                inputs = inputs / tf.norm(inputs, ord=np.inf)

            # broadcast to n_centers x 1 x d
            tf_centers = tf.gather(inputs, tf_center_ids)
            tf_centers = tf.expand_dims(tf_centers, 1)

            # n_centers x n_neighbors x d
            tf_neighbors = tf.gather(inputs, tf_neighbor_ids)

            # compute pairwise distances, then weight, then sum up
            # pairwise is always n_centers x n_neighbors
            if diff_norm == "l1":
                pairwise = tf.reduce_sum(tf.abs(tf_centers - tf_neighbors),
                                         axis=-1)
            elif diff_norm == "l2":
                pairwise = tf.sqrt(
                    tf.reduce_sum((tf_centers - tf_neighbors)**2, axis=-1))
            elif diff_norm == "linf":
                pairwise = tf.reduce_max(tf.abs(tf_centers - tf_neighbors),
                                         axis=-1)
            elif diff_norm == "cos":
                dotprods = tf.reduce_sum(tf_centers * tf_neighbors, axis=-1)
                center_norms = tf.norm(tf_centers, axis=-1)
                neighbor_norms = tf.norm(tf_neighbors, axis=-1)
                # NOTE this computes cosine *similarity* which is why we
                # multiply by -1: minimize the negative similarity!
                cosine_similarity = dotprods / (center_norms * neighbor_norms)
                pairwise = -1 * cosine_similarity
            else:
                raise ValueError("Invalid difference norm specified: {}. "
                                 "Valid are 'l1', 'l2', 'linf', "
                                 "'cos'.".format(weight_norm))
            pairwise_weighted = tf_neighbor_weights * pairwise
            return tf.reduce_sum(pairwise_weighted)

    return neighbor_distance
