import pandas as pd
import math, time, pickle
from helper import *
from test_performance import cluster_test
from transformers import BertTokenizer, BertConfig
from bert_model import BertModel
import mindspore
mindspore.set_context(device_target='GPU', device_id=0)
from mindspore import nn
from find_k_methods import Inverse_JumpsMethod


class BertClassificationModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, target_num, max_length, is_training=True, use_one_hot_embeddings=False):
        super(BertClassificationModel, self).__init__()
        config = BertConfig.from_pretrained('../data/bert-base-uncased')
        config.dtype = mindspore.dtype.float32
        config.compute_type = mindspore.dtype.float16
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0

        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.target_num = target_num
        self.weight_init = mindspore.common.initializer.TruncatedNormal(config.initializer_range)
        self.dense_1 = nn.Dense(config.hidden_size, self.target_num, weight_init=self.weight_init, has_bias=True).to_float(config.compute_type)

        self.cast = mindspore.ops.Cast()
        self.dtype = mindspore.float32

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        # cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        return cls, logits


class BERT_Model(object):

    def __init__(self, params, side_info, input_list, cluster_predict_list, true_ent2clust, true_clust2ent,
                 model_training_time, BERT_self_training_time, sub_uni2triple_dict=None, rel_id2sentence_list=None,
                 K=0):
        self.p = params
        self.side_info = side_info
        self.input_list = input_list
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.model_training_time = model_training_time
        self.BERT_self_training_time = BERT_self_training_time
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.rel_id2sentence_list = rel_id2sentence_list
        self.batch_size = 80
        # self.batch_size = 4
        if self.p.dataset == 'reverb45k_change':
            self.epochs = 100
        else:
            self.epochs = 120
        self.lr = 0.005
        self.K = K
        self.cluster_predict_list = cluster_predict_list
        print('self.epochs:', self.epochs)
        self.coefficient_1, self.coefficient_2 = 0.95, 0.99
        self.max_length = 256

    def fine_tune(self):
        folder = 'multi_view/context_view_' + str(self.p.input)
        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(
            self.model_training_time) + '_' + str(self.BERT_self_training_time)  # for 1

        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        if not checkFile(fname1):
            print('Fine-tune BERT ', 'self.model_training_time:', self.model_training_time,
                  'self.BERT_self_training_time:', self.BERT_self_training_time, fname1)

            target_list = []
            cluster2target_dict = dict()
            num = 0
            for i in range(len(self.cluster_predict_list)):
                label = self.cluster_predict_list[i]
                if label not in cluster2target_dict:
                    cluster2target_dict.update({label: num})
                    num += 1
                target_list.append(cluster2target_dict[label])
            self.target_num = max(target_list) + 1
            self.sentences_list, self.targets_list = [], []
            self.sub2sentence_id_dict = dict()

            print('self.p.input:', self.p.input)
            print('self.max_length:', self.max_length)
            all_length = 0
            num = 0

            for i in range(len(self.input_list)):
                ent_id = self.side_info.ent2id[self.input_list[i]]
                if ent_id in self.side_info.isSub:
                    sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]
                    longest_index, longest_length = 0, 0
                    for j in range(len(sentence_id_list)):
                        id = sentence_id_list[j]
                        sentence = self.side_info.sentence_List[id]
                        if len(sentence) > longest_length and len(sentence) < self.max_length + 50:
                            longest_index, longest_length = j, len(sentence)
                    sentence_id_list = [sentence_id_list[longest_index]]
                    all_length += longest_length
                    sentences_num_list = []
                    for sentence_id in sentence_id_list:
                        sentence = self.side_info.sentence_List[sentence_id]
                        self.sentences_list.append(sentence)
                        target = target_list[i]
                        self.targets_list.append(target)
                        sentences_num_list.append(num)
                        num += 1
                    self.sub2sentence_id_dict.update({i: sentences_num_list})
            ave = all_length / len(self.input_list)
            print('all_length:', all_length, 'ave:', ave)
            print()
            print('self.sentences_list:', type(self.sentences_list), len(self.sentences_list))
            print('self.targets_list:', type(self.targets_list), len(self.targets_list), self.targets_list)
            different_labels = list(set(self.targets_list))
            print('different_labels:', type(different_labels), len(different_labels), different_labels)

            sentence_data = {'sentences': self.sentences_list, 'targets': self.targets_list}
            frame = pd.DataFrame(sentence_data)
            self.sentences = frame['sentences'].values
            self.targets = frame['targets'].values

            self.train_inputs, self.train_targets = self.sentences, self.targets
            batch_count = math.ceil(len(self.train_inputs) / self.batch_size)
            print('batch_count:', batch_count)

            batch_train_inputs, batch_train_targets = [], []
            for i in range(batch_count):
                batch_train_inputs.append(self.train_inputs[i * self.batch_size: (i + 1) * self.batch_size])
                batch_train_targets.append(self.train_targets[i * self.batch_size: (i + 1) * self.batch_size])

            # train the model
            bert_classifier_model = BertClassificationModel(self.target_num, self.max_length)
            optimizer = nn.SGD(bert_classifier_model.get_parameters(), learning_rate=self.lr)
            criterion = nn.CrossEntropyLoss()
            tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
            for epoch in range(self.epochs):
                avg_epoch_loss = 0
                for i in range(batch_count):
                    inputs = batch_train_inputs[i]
                    labels = mindspore.Tensor(batch_train_targets[i], dtype=mindspore.int32)
                    batch_tokenized = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                                  truncation=True,
                                                                  max_length=self.max_length,
                                                                  pad_to_max_length=True)
                    input_ids = mindspore.Tensor(batch_tokenized['input_ids'])
                    attention_mask = mindspore.Tensor(batch_tokenized['attention_mask'])
                    token_type_ids = mindspore.Tensor(batch_tokenized['token_type_ids'])

                    def forward_fn():
                        bert_cls_hidden_state, outputs = bert_classifier_model(input_ids, token_type_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        return loss, bert_cls_hidden_state, outputs

                    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                    (loss, self.bert_cls_hidden_state, outputs), grads = grad_fn()
                    loss = mindspore.ops.depend(loss, optimizer(grads))
                    if epoch == self.epochs - 1:
                        if i == 0:
                            cls_output = self.bert_cls_hidden_state
                            output_label = outputs.argmax(1)
                        else:
                            cls_output = mindspore.ops.concat((cls_output, self.bert_cls_hidden_state), 0)
                            output_label = mindspore.ops.concat((output_label, outputs.argmax(1)), 0)
                    avg_epoch_loss += float(loss.item(0))
                    if i == (batch_count - 1):
                        real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                        print(real_time, "Epoch: %d, Loss: %.4f" % (epoch, avg_epoch_loss))

            self.BERT_CLS = cls_output.asnumpy()
            pickle.dump(self.BERT_CLS, open(fname1, 'wb'))
        else:
            print('load fine-tune BERT CLS  ', 'self.model_training_time:', self.model_training_time,
                  'self.BERT_self_training_time:', self.BERT_self_training_time)
            print('self.BERT_CLS:', fname1)
            self.BERT_CLS = pickle.load(open(fname1, 'rb'))

        print('self.BERT_CLS:', type(self.BERT_CLS), self.BERT_CLS.shape)

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_K_' + str(
            self.BERT_self_training_time)
        if not checkFile(fname3):
            print('Inverse Jump:')
            K_min, K_max = int(self.K * self.coefficient_1), int(self.K * self.coefficient_2)
            gap = int((K_max - K_min) / 20) + 1
            print('K_min:', K_min, 'K_max:', K_max, 'gap:', gap)
            cluster_list = range(K_min, K_max, gap)
            jm = Inverse_JumpsMethod(data=self.BERT_CLS, k_list=cluster_list, dim_is_bert=True)
            jm.Distortions(random_state=0)
            distortions = jm.distortions
            jm.Jumps(distortions=distortions)
            level_one_Inverse_JumpsMethod = jm.recommended_cluster_number
            pickle.dump(level_one_Inverse_JumpsMethod, open(fname3, 'wb'))
        else:
            print('load level_one_Inverse_JumpsMethod:', fname3)
            level_one_Inverse_JumpsMethod = pickle.load(open(fname3, 'rb'))

        print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)

        dist = pdist(self.BERT_CLS, metric=self.p.metric)
        clust_res = linkage(dist, method=self.p.linkage)
        clusters = fcluster(clust_res, t=level_one_Inverse_JumpsMethod, criterion='maxclust') - 1
        cluster_predict_list = list(clusters)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                           self.true_clust2ent)
        print('self.model_training_time:', self.model_training_time,
              'self.BERT_self_training_time:', self.BERT_self_training_time, 'Best BERT CLS result:')
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()
        return cluster_predict_list, level_one_Inverse_JumpsMethod
