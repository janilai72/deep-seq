#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2019/10/05 03:18:45
@Author  :   Jian.Lai
@Version :   1.0
@Contact :   jani.lai@aliyun.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   源码demo： https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/crf
'''
import numpy as np
import tensorflow as tf
 
# 参数设置
batch_zise = 10
num_words = 20
num_features = 100
num_tags = 7
 
# 构建随机特征
x = np.random.rand(batch_zise, num_words, num_features).astype(np.float32)
 
# 构建随机tag
y = np.random.randint(
    num_tags, size=[batch_zise, num_words]).astype(np.int32)
 
# 获取样本句长向量（因为每一个样本可能包含不一样多的词），在这里统一设为 num_words - 1，真实情况下根据需要设置
sequence_lengths = np.full(batch_zise, num_words - 1, dtype=np.int32)
print('x:',x)
print('y:',y)
print('sequence_lengths:',sequence_lengths)
# 训练，评估模型
with tf.Graph().as_default():
    with tf.Session() as session:
        x_t = tf.constant(x)
        y_t = tf.constant(y)
        sequence_lengths_t = tf.constant(sequence_lengths)
 
        # 在这里设置一个无偏置的线性层
        weights = tf.get_variable("weights", [num_features, num_tags])
        matricized_x_t = tf.reshape(x_t, [-1, num_features])
        print(matricized_x_t.shape)
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [batch_zise, num_words, num_tags])

        print(unary_scores.shape)
        print(y_t.shape)
        print(sequence_lengths_t.shape)
        # 计算log-likelihood并获得transition_params
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            unary_scores, y_t, sequence_lengths_t)
 
        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            unary_scores, transition_params, sequence_lengths_t)
 
        loss = tf.reduce_mean(-log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
 
        session.run(tf.global_variables_initializer())
 
        mask = (np.expand_dims(np.arange(num_words), axis=0) <     # np.arange()创建等差数组
                np.expand_dims(sequence_lengths, axis=1))          # np.expand_dims()扩张维度
 
        # 得到一个batch_zise*num_words的二维数组，数据类型为布尔型，目的是对句长进行截断
 
        # 将每个样本的sequence_lengths加起来，得到标签的总数
        total_labels = np.sum(sequence_lengths)
 
        # 进行训练
        for i in range(1000):
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
            if i % 100 == 0:
                correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)
