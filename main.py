from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn import preprocessing


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def extract(input, n_fea, time_window, moving):
    global n_classes
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

def compute_accuracy_t(v_xs, v_ys):  # this function only calculate the acc of CNN_task
    global prediction_t
    y_pre = sess.run(prediction_t, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys_t: v_ys, keep_prob: keep})
    return result


def compute_accuracy_p(v_xs, v_ys):  # this function only calculate the acc of CNN_task
    global prediction_p
    y_pre = sess.run(prediction_p, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys_p: v_ys, keep_prob: keep})
    return result


# # WoW! use this to limit the GPU number
# import os
# GPU_ID = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
# print('Let`s start!, GPU:', GPU_ID)

################THU seizure data reading
# thr first 21 columns are features, the 22th column is serzure/normal, the 23th column is person label.
# in the task lable, 0: normal, 1: seizure

# data reading
# python 3: add ',encoding='iso-8859-1'' in the pickle.load.
all_data = pickle.load(open("/home/xiangzhang/scratch/all_14sub.p", "rb" ))
print(type(all_data), all_data.shape, all_data[:, -1])

n_classes = 2
n_person_ = 13  # the number of training subjects
sample_persub = 250*500  # we have overlapping now
print(type(all_data), all_data.shape, all_data[:, -1])

no_fea = 21  # data.shape[-1] - 1
seg_length = 250  # # 255 for raw data, 96 for layer 23, 64 for layer 2, 32 for layer 2

scaler = preprocessing.MinMaxScaler()  # normalization
F = scaler.fit_transform(all_data[:, :no_fea])  # scale to [0, 1]

all_data = np.hstack((F, all_data[:, no_fea:no_fea+1]))  # only use the task ID


"""Make person label"""
n_sample_ = int(2*sample_persub/seg_length )  # the number of sampls of each subject after reshape
ll = np.ones([n_sample_, 1])*0
for hh in range(1, n_person_):
    ll_new = np.ones([n_sample_, 1])*hh
    ll = np.vstack((ll, ll_new))
print('the shape of maked person label', ll.shape)

ll_test = np.ones([n_sample_, 1])*n_person_

ss_train = time.clock()
# Person Independent
for P_ID in range(14):  # n_person_++1
    if P_ID==0:
        reuse=False
    else:
        reuse=True
    """Select train and test subject"""
    data_ = all_data[sample_persub*P_ID:sample_persub*(P_ID+1)]

    list = range(sample_persub*P_ID, sample_persub*(P_ID+1))
    data = np.delete(all_data, list, axis=0)
    # overlap
    train_data = extract(data, n_fea=no_fea, time_window=seg_length, moving=(seg_length/2))
    test_data = extract(data_, n_fea=no_fea, time_window=seg_length, moving=(seg_length/2))  # 50% overlapping
    # continue
    """Replace the original person data by the maked data"""
    no_fea_long = train_data.shape[-1] - 1  # here is - 2, because has two IDs
    print(train_data[:, :no_fea_long+1].shape, ll.shape)
    train_data = np.hstack((train_data[:, :no_fea_long+1], ll))
    test_data = np.hstack((test_data[:, :no_fea_long + 1], ll_test))
    print(train_data.shape)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    print(train_data.shape, test_data.shape)

    feature_train = train_data[:, :no_fea_long]
    feature_test = test_data[:, :no_fea_long]
    label_train_t = train_data[:, no_fea_long:no_fea_long + 1]
    label_test_t = test_data[:, no_fea_long:no_fea_long + 1]
    label_train_p = train_data[:, no_fea_long + 1:no_fea_long + 2]

    label_train_t = one_hot(label_train_t)
    label_test_t = one_hot(label_test_t)
    label_train_p = one_hot(label_train_p)


    n_class_t = 2  # 0-3
    n_class_p = n_person_  # 0-8
    keep = 0.8

    a = feature_train

    ## batch split
    batch_size = int(feature_test.shape[0])
    train_fea = []
    n_group = int(feature_train.shape[0]/feature_test.shape[0])
    for i in range(n_group):
        f = a[(0+batch_size*i):(batch_size+batch_size*i)]
        train_fea.append(f)
    print (train_fea[0].shape)

    train_label_t=[]
    for i in range(n_group):
        f = label_train_t[(0 + batch_size * i):(batch_size + batch_size * i), :]
        train_label_t.append(f)
    print (train_label_t[0].shape)

    train_label_p = []
    for i in range(n_group):
        f = label_train_p[(0 + batch_size * i):(batch_size + batch_size * i), :]
        train_label_p.append(f)
    print (train_label_p[0].shape)

    """Placeholder"""
    # define placeholder for inputs to network
    xs = tf.compat.v1.placeholder(tf.float32, [None, no_fea_long], name='xsss')  # 249*100
    ys_t = tf.compat.v1.placeholder(tf.float32, [None, n_class_t], name='ys_t')
    ys_p = tf.compat.v1.placeholder(tf.float32, [None, n_class_p], name='ys_p')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep')

    """AE code, whihc is divided into two represents"""
    """Use tf.nn.relu in the hidden layer if maxmin scaler; sigmoid if z-score;
    use maxmin, AE converge better but the classification training acc cannot reach 100%;
    use z-score, the opposite. I perfer z-score. Or use maxmin, make the network deeper?
    """
    """Convolutional AE"""
    with tf.compat.v1.variable_scope("AE", reuse=reuse):
        # dim_code = 1000
        input = tf.reshape(xs, [-1, seg_length, no_fea, 1])  # [200, 14]
        input = tf.contrib.layers.batch_norm(input, decay=0.9)
        input = tf.nn.dropout(input, keep_prob)
        l_AE, w_AE = 2, 1
        print(xs.shape)  # [n_samples, 28,28,1]

        depth_AE = 4  # default is 8
        conv1 = tf.layers.conv2d(inputs=input, filters=depth_AE, kernel_size=[2, 2], padding="same",
                                 activation=tf.nn.relu)
        h_t = tf.layers.max_pooling2d(inputs=conv1, pool_size=[l_AE, w_AE], strides=[l_AE, w_AE])
        # pool1 = tf.contrib.layers.batch_norm(pool1, decay=0.9)

        conv1_p = tf.layers.conv2d(inputs=input, filters=depth_AE, kernel_size=[2, 2], padding="same",
                                   activation=tf.nn.relu)
        h_p = tf.layers.max_pooling2d(inputs=conv1_p, pool_size=[l_AE, w_AE], strides=[l_AE, w_AE])

        # decoder
        output_t = tf.layers.conv2d_transpose(h_t, kernel_size=5, filters=1, strides=[l_AE, w_AE], padding='same')
        # output_t = tf.nn.relu(tf.contrib.layers.batch_norm(output_t, decay=0.9))
        output_p = tf.layers.conv2d_transpose(h_p, kernel_size=5, filters=1, strides=[l_AE, w_AE], padding='same')
        # output_p = tf.nn.relu(tf.contrib.layers.batch_norm(output_p, decay=0.9))
        output = (output_t + output_p) / 2

        # #another decoder
        # h = (h_t + h_p)/2
        # output = tf.layers.conv2d_transpose(h, kernel_size=5, filters=1, strides=[l_AE, w_AE], padding='same')
        # output = tf.nn.relu(tf.contrib.layers.batch_norm(output, decay=0.9))

        output = tf.reshape(output, [-1, seg_length * no_fea])

    """CNN code for task, maybe we can make it deeper? """
    l_1, w_1 = 2, 2
    l_2, w_2 = 2, 2
    l_3, w_3 = 2, 2
    l_4, w_4 = 2, 1

    with tf.compat.v1.variable_scope("class_t", reuse=reuse):
        # x_image_t = tf.reshape(h_t, [-1, 10, 10, 1])  # [200, 14]
        x_image_t = tf.contrib.layers.batch_norm(h_t, decay=0.9)

        # x_image_t = tf.nn.dropout(x_image_t, keep_prob)
        print(x_image_t.shape)  # [n_samples, 28,28,1]
        depth_1 = 16  # default is 8
        conv1 = tf.layers.conv2d(inputs=x_image_t, filters=depth_1, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[l_1, w_1], strides=[l_1, w_1])
        pool1 = tf.contrib.layers.batch_norm(pool1, decay=0.9)

        depth_2 = 32  # default is 32
        conv2 = tf.layers.conv2d(inputs=pool1, filters=depth_2, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[l_2, w_2], strides=[l_2, w_2])
        pool2 = tf.contrib.layers.batch_norm(pool2, decay=0.9)

        depth_3 = 64
        conv3 = tf.layers.conv2d(inputs=pool2, filters=depth_3, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[l_3, w_3], strides=[l_3, w_3])
        pool3 = tf.contrib.layers.batch_norm(pool3, decay=0.9)
        # print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),)

        depth_4 = 128
        conv4 = tf.layers.conv2d(inputs=pool3, filters=depth_4, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[l_4, w_4], strides=[l_4, w_4])
        pool4 = tf.contrib.layers.batch_norm(pool4, decay=0.9)

        fc1 = tf.contrib.layers.flatten(pool4)  # flatten the pool 2
        print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),  pool4.get_shape(), fc1.get_shape())


        # """Add another FC layer"""
        fc1 = tf.layers.dense(fc1, units=300, activation=tf.nn.sigmoid)
        fc1 = tf.nn.dropout(fc1, keep_prob)

        dim_hidden = 21
        fc3 = tf.layers.dense(fc1, units=dim_hidden, activation=tf.nn.sigmoid)
        fc3 = tf.nn.dropout(fc3, keep_prob)

        # Attention layer
        att = tf.layers.dense(xs, units=fc3.shape[-1], activation=tf.nn.sigmoid)
        fc3 = tf.multiply(fc3, att)

        prediction_t = tf.layers.dense(fc3, units=n_class_t, activation=None)
        print('prediction_t', prediction_t.get_shape())

    """CNN code for person"""
    with tf.compat.v1.variable_scope("class_p", reuse=reuse):
        x_image_p = tf.contrib.layers.batch_norm(h_p, decay=0.9)

        conv1_p = tf.layers.conv2d(inputs=x_image_p, filters=depth_1, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1_p = tf.layers.max_pooling2d(inputs=conv1_p, pool_size=[l_1, w_1], strides=[l_1, w_1])
        pool1_p = tf.contrib.layers.batch_norm(pool1_p, decay=0.9)

        conv2_p = tf.layers.conv2d(inputs=pool1_p, filters=depth_2, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2_p = tf.layers.max_pooling2d(inputs=conv2_p, pool_size=[l_2, w_2], strides=[l_2, w_2])
        pool2_p = tf.contrib.layers.batch_norm(pool2_p, decay=0.9)

        conv3_p = tf.layers.conv2d(inputs=pool2_p, filters=depth_2, kernel_size=[2, 2], padding="same",
                                   activation=tf.nn.relu)
        pool3_p = tf.layers.max_pooling2d(inputs=conv3_p, pool_size=[l_3, w_3], strides=[l_3, w_3])
        pool3_p = tf.contrib.layers.batch_norm(pool3_p, decay=0.9)

        conv4_p = tf.layers.conv2d(inputs=pool3_p, filters=depth_4, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)
        pool4_p = tf.layers.max_pooling2d(inputs=conv4_p, pool_size=[l_4, w_4], strides=[l_4, w_4])
        pool4_p = tf.contrib.layers.batch_norm(pool4_p, decay=0.9)

        fc1_p = tf.contrib.layers.flatten(pool4_p)  # flatten the pool 2

        dim_hidden_p = 200
        fc3_p = tf.layers.dense(fc1_p, units=dim_hidden_p, activation=tf.nn.sigmoid)

        fc3_p = tf.nn.dropout(fc3_p, keep_prob)

        prediction_p = tf.layers.dense(fc3_p, units=n_class_p, activation=None)
        print('prediction_p', tf.shape(prediction_p))

    def kl_divergence(p, q):
        return tf.reduce_sum(p * tf.log(p/q))


    """cost calculation"""
    train_vars = tf.trainable_variables()
    l2_AE = 0.005 * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.startswith("AE"))
    l2_class = 0.005 * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.startswith("class"))
    """multiply 5 to enhance the cross_entropy_t """
    cross_entropy_t = 10*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_t, labels=ys_t))
    cross_entropy_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_p, labels=ys_p))
    """Add 0.1 to reduce the cost_AE"""
    cost_AE = tf.reduce_mean(tf.pow(xs - output, 2)) + l2_AE

    class_vars = [var for var in train_vars if var.name.startswith("class")]  # discriminator tensor
    AE_vars = [var for var in train_vars if var.name.startswith("AE")]
    t_vars = [var for var in train_vars if var.name.startswith("class_t")]

    cost = cost_AE  + cross_entropy_t +   cross_entropy_p + l2_class + l2_AE
    lr = 0.00005  # use 0.0001 for parameter tuning
    with tf.compat.v1.variable_scope("optimization", reuse=reuse):
        train_step_task = tf.train.AdamOptimizer(lr).minimize(cost, )
        # train_step_AE = tf.train.AdamOptimizer(lr).minimize(cost_AE+l2_AE, var_list=AE_vars)
        train_step_t = tf.train.AdamOptimizer(lr).minimize(cross_entropy_t)
    # 1. AE learning rate= 0.00001 2. dim_code larger better, 3. add cost_AE on cost.

    con = tf.ConfigProto()
    con.gpu_options.allow_growth = True
    sess = tf.Session(config=con)
    init = tf.global_variables_initializer()
    sess.run(init)
    # History records of loss functions
    cost_his = []
    cost_AE_his = []
    cost_t_his = []
    cost_p_his = []

    test_cost_t_his = []

    start=time.clock()
    step = 1
    while step < 51:  # 251 iterations
        print('iteration step', step)
        for i in range(n_group):
            feed = {xs: train_fea[i], ys_t: train_label_t[i], ys_p: train_label_p[i], keep_prob:keep}
            sess.run(train_step_task, feed_dict=feed)
            sess.run(train_step_t, feed_dict=feed)

        if step % 10 == 0:
            """training cost"""
            cost_, cost_AE_, cross_entropy_p_, cross_entropy_t_=sess.run([cost, cost_AE, cross_entropy_p, cross_entropy_t],
                                                                         feed_dict={xs: train_fea[0],
                                             ys_t: train_label_t[0], ys_p: train_label_p[0], keep_prob:keep})

            """testing cost"""
            cost_AE_test_, cross_entropy_t_test_=sess.run([cost_AE, cross_entropy_t],
                                                          feed_dict ={xs: feature_test, ys_t: label_test_t, keep_prob: keep})

            print('person, step:',P_ID, step, 'train acc task', compute_accuracy_t(feature_train, label_train_t),
                  'train acc person', compute_accuracy_p(feature_train, label_train_p),
                  ',the test acc task', compute_accuracy_t(feature_test, label_test_t),
                  'testing: AE, t', cost_AE_test_, cross_entropy_t_test_)

            print('training cost: total, AE, t, p',cost_, cost_AE_,  cross_entropy_t_, cross_entropy_p_)
            cost_his.append(cost_)
            cost_AE_his.append(cost_AE_)
            cost_t_his.append(cross_entropy_t_)
            cost_p_his.append(cross_entropy_p_)

            test_cost_t_his.append(cross_entropy_t_test_)

        # save the attention weights for fine-grained analysis
        if step % 50 == 0:
            att_ = sess.run(att, feed_dict={xs: feature_test, ys_t: label_test_t, keep_prob: keep})

            ss = time.clock()
            pred = sess.run(prediction_t, feed_dict={xs: feature_test, ys_t: label_test_t, keep_prob: keep})
            print('training, testing time', time.clock()-ss_train, time.clock()-ss)

            pickle.dump(att_,  open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/TUH_attention_P'
                                    +str(step)+'_backup.p',  "wb"), protocol=2)
            print('attention saved, person:', P_ID)

        step += 1

    # save the cost history values for convergence analysis
    pickle.dump(cost_his, open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/cost_his.p', "wb"))
    pickle.dump(cost_AE_his,
                open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/cost_AE_his.p', "wb"))
    pickle.dump(cost_t_his,
                open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/cost_t_his.p', "wb"))
    pickle.dump(cost_p_his,
                open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/cost_p_his.p', "wb"))
    pickle.dump(test_cost_t_his,
                open('/home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure/test_cost_t_his.p', "wb"))
    print("five losses are saved at /home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure")




