#!/usr/bin/env/python
'''
Usage:
    adap_gcn_cv_samegate.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir=NAME           log dir name
    --data_dir=NAME          data dir name
    --restore=FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --data_type=TYPE         grounded or lifted
    --index_type=TYPE        random or domain
    --save_dir=NAME          save dir name
    --random_seed=K          random seed [default: 123]
    --learning_rate=LR       learning rate
    --num_timesteps=N        num_timesteps
    --hidden_size=H          hidden size
    --split_index=M          index of cross-validation
'''
from typing import Tuple, Sequence, Any

from docopt import docopt
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import sys, traceback
import pdb

from adap_base_cv import ChemModel
from utils import glorot_init, one_hot

import os
import psutil

def print_mem(name):
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print("memory use:",name, memoryUse)


class asSparseGCNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({'batch_size': 1000000,
                       'task_sample_ratios': {},
                       'gcn_use_bias': False,
                       'graph_state_dropout_keep_prob': 1.0,
                       })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_list'] = tf.placeholder(tf.int64, [None, 2], name='adjacency_list')
        self.placeholders['adjacency_weights'] = tf.placeholder(tf.float32, [None], name='adjacency_weights')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')

        with tf.variable_scope('gcn_scope'):
            self.weights['edge_weights'] = [tf.Variable(glorot_init((h_dim, h_dim)), name="gcn_weights_%i" % i)
                                            for i in range(self.params['num_timesteps'])]

            if self.params['gcn_use_bias']:
                self.weights['edge_biases'] = [tf.Variable(np.zeros([h_dim], dtype=np.float32), name="gcn_bias_%i" % i)
                                               for i in range(self.params['num_timesteps'])]

    def compute_final_node_representations(self):
        with tf.variable_scope('gcn_scope'):
            cur_node_states = self.placeholders['initial_node_representation']  # number of nodes in batch v x D
            num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int64)[0]

            adjacency_matrix = tf.SparseTensor(indices=self.placeholders['adjacency_list'],
                                               values=self.placeholders['adjacency_weights'],
                                               dense_shape=[num_nodes, num_nodes])

            for layer_idx in range(self.params['num_timesteps']):
                scaled_cur_node_states = tf.sparse_tensor_dense_matmul(adjacency_matrix, cur_node_states)  # v x D
                new_node_states = tf.matmul(scaled_cur_node_states, self.weights['edge_weights'][layer_idx])

                if self.params['gcn_use_bias']:
                    new_node_states += self.weights['edge_biases'][layer_idx]  # v x D

                # On all but final layer do ReLU and dropout:
                if layer_idx < self.params['num_timesteps'] - 1:
                    new_node_states = tf.nn.relu(new_node_states)
                    new_node_states = tf.nn.dropout(new_node_states, keep_prob=self.placeholders['graph_state_keep_prob'])

                cur_node_states = new_node_states

            return cur_node_states

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)    # [v x 1]

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]
        return tf.squeeze(graph_representations)  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool, context_file=None, batch_size=None) -> Any:
        processed_graphs = []
        # tmp=[]
        for d in raw_data:
            (adjacency_list, adjacency_weights) = self.__graph_to_adjacency_list(d['graph'], min(len(d["node_features"]), batch_size))
            # tmp.append([float(d["targets"][task_id][0]) for task_id in self.params['task_ids']])
            # print(len(d["node_features"]))
            init_feature = d["node_features"]
            if batch_size < len(d["node_features"]):
                init_feature = init_feature[:batch_size]
            processed_graphs.append({"adjacency_list": adjacency_list,
                                     "adjacency_weights": adjacency_weights,
                                     "init": init_feature,
                                     "labels": [float(d["targets"][task_id][0]) for task_id in self.params['task_ids']]})
        # tmp = np.array(tmp)
        # tmp1 = sum(tmp>1800)
        # tmp2 = sum(tmp>900)
        # print(tmp1)
        # print(tmp2)

        pair_graphs = []
        skipped_ids = []
        if is_training_data:
            np.random.shuffle(processed_graphs)
            # for task_id in self.params['task_ids']:
            #     task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
            #     if task_sample_ratio is not None:
            #         ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
            #         for ex_id in range(ex_to_sample, len(processed_graphs)):
            #             processed_graphs[ex_id]['labels'][task_id] = None
        if context_file is None:
            for ex_id in range(len(processed_graphs)):#generating new training data
                preplanner_ids = np.where(np.array(processed_graphs[ex_id]["labels"]) > self.params['T']/2.0)[0]
                for pre_id in preplanner_ids:
                    currentlabels = processed_graphs[ex_id]["labels"]
                    newlabels = [x > self.params['T']/2.0 for x in currentlabels]
                    newlabels[pre_id] = currentlabels[pre_id]>self.params['T']


                    pair_graphs.append({"adjacency_list": processed_graphs[ex_id]["adjacency_list"],
                                         "adjacency_weights": processed_graphs[ex_id]["adjacency_weights"],
                                         "init": processed_graphs[ex_id]["init"],
                                         "labels": newlabels,
                                        "pre_id": pre_id})
        else:
            cont_values = np.load(open(context_file,"rb"))
            preplanner_ids = np.argmin(cont_values, axis=1)
            for ex_id in range(len(processed_graphs)):
                pre_id = preplanner_ids[ex_id]
                pre_label = processed_graphs[ex_id]["labels"][pre_id]
                if pre_label < self.params['T']/2.0:
                    skipped_ids.append(ex_id)
                    continue
                else:
                    currentlabels = processed_graphs[ex_id]["labels"]
                    newlabels = [x > self.params['T'] / 2.0 for x in currentlabels]
                    newlabels[pre_id] = currentlabels[pre_id] > self.params['T']
                    pair_graphs.append({"adjacency_list": processed_graphs[ex_id]["adjacency_list"],
                                        "adjacency_weights": processed_graphs[ex_id]["adjacency_weights"],
                                        "init": processed_graphs[ex_id]["init"],
                                        "labels": newlabels,
                                        "graph_id":ex_id,
                                        "pre_id": pre_id})
        return pair_graphs, skipped_ids

    def __graph_to_adjacency_list(self, graph, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        # # Step 1: Generate adjacency matrices:
        # adj_matrix = np.zeros((num_nodes, num_nodes))
        # for src, _, dest in graph:
        #     adj_matrix[src, dest] = 1
        #     adj_matrix[dest, src] = 1
        #
        # # Step 2: Introduce self loops:
        # self_loops = np.eye(num_nodes)
        # adj_matrix += self_loops
        #
        #
        # # Step 3: Normalize adj_matrices so that scale of vectors doesn't explode:
        # row_sum = np.sum(adj_matrix, axis=-1)
        # D_inv_sqrt = np.diag(np.power(row_sum, -0.5).flatten() + 1e-7)
        # adj_matrix = D_inv_sqrt.dot(adj_matrix).dot(D_inv_sqrt)

        adj_matrix = sp.dok_matrix((num_nodes, num_nodes))
        for src, _, dest in graph:
            if src < num_nodes and dest < num_nodes:
                adj_matrix[src, dest] = 1
                adj_matrix[dest, src] = 1
        adj_matrix = adj_matrix.tocsr()
        self_loops = sp.eye(num_nodes)
        adj_matrix += self_loops

        row_sum = adj_matrix.sum(axis=-1).A
        D_inv_sqrt = sp.diags((np.power(row_sum, -0.5).flatten() + 1e-7))
        adj_matrix = D_inv_sqrt.dot(adj_matrix).dot(D_inv_sqrt)

        row, col = adj_matrix.nonzero()
        final_adj_weights = adj_matrix.data
        final_adj_list = []
        final_adj_list.append(row)
        final_adj_list.append(col)

        # Step 4: Turn into sorted adjacency lists:
        # adj_matrix = adj_matrix.tolil()
        # final_adj_list = []
        # final_adj_weights = []
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         w = adj_matrix[i, j]
        #         if w != 0:
        #             final_adj_list.append([i,j])
        #             final_adj_weights.append(w)
        # print_mem("2_")
        return np.array(final_adj_list).T , np.array(final_adj_weights)

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            np.random.shuffle(data)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_task_mask = []
            batch_adjacency_list = []
            batch_adjacency_weights = []
            batch_graph_nodes_list = []
            batch_next_id_vectors = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) <= self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         mode='constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                batch_adjacency_list.append(cur_graph['adjacency_list'] + node_offset)
                batch_adjacency_weights.append(cur_graph['adjacency_weights'])
                batch_next_id_vectors.append(one_hot(len(self.params['task_ids']),cur_graph['pre_id']))

                target_task_values = []
                target_task_mask = []
                for target_val in cur_graph['labels']:
                    if target_val is None:  # This is one of the examples we didn't sample...
                        target_task_values.append(0.)
                        target_task_mask.append(0.)
                    else:
                        target_task_values.append(target_val)
                        target_task_mask.append(1.)
                batch_target_task_values.append(target_task_values)
                batch_target_task_mask.append(target_task_mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph


            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['adjacency_list']: np.concatenate(batch_adjacency_list, axis=0),
                self.placeholders['adjacency_weights']: np.concatenate(batch_adjacency_weights, axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list, axis=0),
                self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1,0]),
                self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['pre_id_vector']:batch_next_id_vectors
            }

            yield batch_feed_dict


def main():
    args = docopt(__doc__)
    try:
        model = asSparseGCNChemModel(args)

        if '--data_type' in args and args['--data_type'] is not None:
            data_type = args['--data_type']
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        test_file = data_dir + '/data-' + data_type + '-2018-07-20-test.json'
        if '--save_dir' in args and args['--save_dir'] is not None:
            save_dir = args['--save_dir']
        else:
            save_dir = "./"
        if '--split_index' in args and args['--split_index'] is not None:
            split_index = int(args['--split_index'])
            model.train(split_index)
            # model.restore_model("models/lifted/2018-07-29-01-12-46_4493_asSparseGGNNChemModel_model_lifted.pickle")
            contextfile = os.path.join(save_dir, 'planner1_gcn_samegate_' + data_type + '_' + str(split_index) + '.npz')
            timeout, _, preds, skipped_graphs = model.pred(test_file, contextfile=contextfile)
            savefile = os.path.join(save_dir, 'planner2_gcn_samegate_' + data_type + '_' + str(split_index) + '.npz')
            print("Split: ", split_index, "Timeout number,", timeout)
            np.savez(savefile, preds=preds, skipped=skipped_graphs)
        else:
            for i in range(10):
                model.train(i)

                # model.restore_model("models/grounded/2018-07-29-22-50-23_11345_asSparseGCNChemModel_model_best.pickle")
                contextfile = os.path.join(save_dir, 'planner1_gcn_samegate_' + data_type + '_' + str(i) + '.npz')
                if not os.path.exists(contextfile):
                    continue
                timeout,_,preds, skipped_graphs = model.pred(test_file,contextfile = contextfile)
                savefile = os.path.join(save_dir, 'planner2_gcn_samegate_' + data_type + '_' + str(i) + '.npz')
                print("Split: ", i, "Timeout number,", timeout)
                # preds, acc = model.test('./data/masterplan/masterplan_test_subsample.json')
                # # preds, acc = model.test('./data/masterplan_test.json')
                np.savez(savefile, preds=preds, skipped=skipped_graphs)
                # print("GCN Test Accuracy,", acc)

    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    main()
