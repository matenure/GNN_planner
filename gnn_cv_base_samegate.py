#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random

from utils import MLP, ThreadedIterator, SMALL_NUMBER


class ChemModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 300,
            'patience': 10,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': list(range(17)),

            'random_seed': 277,

            'batch_size': 100000

            # 'train_file': 'data/masterplan_new/data-lifted-2018-07-20-train.json',
            # 'valid_file': 'data/masterplan_new/data-lifted-2018-07-20-valid.json'

            # 'train_file': 'data/masterplan/masterplan_train_subsample.json',
            # 'valid_file': 'data/masterplan/masterplan_valid_subsample.json'
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        data_file = ""
        if '--data_type' in args and args['--data_type'] is not None:
            data_type = args['--data_type']
            data_file = "data-" + data_type + "-2018-07-20-train-valid.json"
        self.data_file = data_file

        index_dir = ""
        if '--index_type' in args and args['--index_type'] is not None:
            index_type = args['--index_type']
            if index_type =="random":
                index_dir = "random-idx"
                train_index_suffix = "-train-split-rnd-2018-07-20.txt"
                valid_index_suffix = "-valid-split-rnd-2018-07-20.txt"
            if index_type == "domain":
                index_dir = "domain-preserving-idx"
                train_index_suffix = "-train-split-domains-2018-07-20.txt"
                valid_index_suffix = "-valid-split-domains-2018-07-20.txt"
        self.index_dir = os.path.join(self.data_dir, index_dir)
        self.train_index = []
        self.valid_index = []
        for i in range(10):
            train_index_file = os.path.join(self.index_dir, str(i) + train_index_suffix)
            self.train_index.append(self.load_index_file(train_index_file))
            valid_index_file = os.path.join(self.index_dir, str(i) + valid_index_suffix)
            self.valid_index.append(self.load_index_file(valid_index_file))


        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid()),self.__class__.__name__])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))

        random_seed = args.get("--random_seed")
        if random_seed is not None:
            params["random_seed"] = int(random_seed)

        learning_rate = args.get("--learning_rate")
        if learning_rate is not None:
            params["learning_rate"] = float(learning_rate)

        num_timesteps = args.get("--num_timesteps")
        if num_timesteps is not None:
            params["num_timesteps"] = int(num_timesteps)

        hidden_size = args.get("--hidden_size")
        if hidden_size is not None:
            params["hidden_size"] = int(hidden_size)

        params["data_file"] = data_file
        params["index_dir"] = index_dir

        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])


        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.all_data = self.load_data(params['data_file'], is_training_data=False) # no shuffle

        # self.train_data = self.load_data(params['train_file'], is_training_data=True)
        # self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data, batch_size=self.params['batch_size'])

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, batch_size=None) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(preds>=0.5, labels>=0.5)
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        self.ops['predicted_values'] = []

        with tf.variable_scope("regression_gate"):
            self.weights['regression_gate_task'] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                   self.placeholders['out_layer_dropout_keep_prob'])

        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                # with tf.variable_scope("regression_gate"):
                #     self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                #                                                            self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task'],
                                                        self.weights['regression_transform_task%i' % task_id])
                predictions = tf.nn.sigmoid(computed_values)
                self.ops['predicted_values'].append(predictions)
                diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.placeholders['target_values'][internal_id,:],logits=computed_values)
                task_target_mask = self.placeholders['target_mask'][internal_id,:]
                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
                diff = diff * task_target_mask  # Mask out unused values

                self.ops['accuracy_task%i' % task_id] = self.masked_accuracy(predictions,
                                                        self.placeholders['target_values'][internal_id,:], task_target_mask)

                task_loss = tf.reduce_sum(diff) / task_target_num
                # Normalise loss to account for fewer task-specific examples in batch:
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['losses'].append(task_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        loss = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            # print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
            #                                                                    step,
            #                                                                    num_graphs,
            #                                                                    loss / processed_graphs),
            #       end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        # error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        error_ratios = accuracies
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec

    def test(self, testfile):
        if os.path.exists(self.best_model_file):
            self.restore_model(self.best_model_file)
        testdata = self.load_data(testfile,is_training_data=False)
        processed_graphs = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(testdata, False), max_queue_size=5)
        preds = []
        accs = []
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs

            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['predicted_values'], accuracy_ops]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_pred, batch_accuracies) = (result[0], result[1])
            batch_pred = np.array(batch_pred)
            if len(batch_pred.shape) == 1:
                batch_pred = np.expand_dims(batch_pred, 1)
            preds.append(batch_pred.T)
            accs.append(np.array(batch_accuracies)*num_graphs)
        return np.concatenate(preds,0), np.sum(accs, axis=0)/float(processed_graphs)

    def pred(self, testfile):#output the planner with the lowest prob to be timeout, and get the percentage of timeout using these planners
        if os.path.exists(self.best_model_file):
            self.restore_model(self.best_model_file)
        testdata = self.load_data(testfile,is_training_data=False)
        processed_graphs = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(testdata, False), max_queue_size=5)
        preds = []
        accs = []
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs

            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['predicted_values'], accuracy_ops]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_pred, batch_accuracies) = (result[0], result[1])
            batch_pred = np.array(batch_pred)
            if len(batch_pred.shape)==1:
                batch_pred = np.expand_dims(batch_pred,1)

            preds.append(batch_pred.T)
            accs.append(np.array(batch_accuracies)*num_graphs)
        planner = np.argmin(np.concatenate(preds,0),axis=1)
        pred_labels = [testdata[ex_id]["labels"][planner[ex_id]] for ex_id in range(len(testdata))]
        return np.concatenate(preds,0), sum(pred_labels), len(testdata)

    def load_index_file(self, indexfile):
        with open(indexfile, 'r') as f:
            indecies = f.readlines()
        indecies = [int(a.strip()) for a in indecies]
        return np.array(indecies)

    def train(self, index):
        log_to_save = []
        total_time_start = time.time()
        current_train_index = np.random.permutation(self.train_index[index])
        current_valid_index = np.random.permutation(self.valid_index[index])
        self.train_data = [self.all_data[i] for i in current_train_index]
        self.valid_data = [self.all_data[i] for i in current_valid_index]


        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                valid_loss, valid_accs, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_loss = valid_loss
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_loss, best_val_acc_epoch) = (0, float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_accs, train_errs, train_speed = self.run_epoch("epoch %i (training)" % epoch,
                                                                                 self.train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s  | instances/sec: %.2f" % (train_loss, accs_str, train_speed))
                valid_loss, valid_accs, valid_errs, valid_speed = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                 self.valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s  | instances/sec: %.2f" % (valid_loss, accs_str, valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                val_acc = np.sum(valid_accs)  # type: float
                if valid_loss <best_val_loss:
                # if val_acc > best_val_acc:
                    self.save_model(self.best_model_file)
                    # print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_loss = valid_loss
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
                         "params": self.params,
                         "weights": weights_to_save
                       }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs','train_file', 'valid_file']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            if len(variables_to_initialize)>0:
                restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
