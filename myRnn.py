import numpy as np

def pro_data(file,nums=50):
    with open(file,encoding="utf-8") as f:
        datas = f.read().split("\n")[:nums]

    word_2_index = {}
    index_2_word = []

    for article in datas:
        for w in article:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
                index_2_word.append(w)

    return datas,word_2_index,index_2_word,np.eye(len(word_2_index))

def make_x_y(poetry):
    global word_2_index, index_2_word, wordidx_2_onehot

    w_onehot = [wordidx_2_onehot[word_2_index[i]] for i in poetry]

    x = w_onehot[:-1]
    y = w_onehot[1:]
    return x,y

def generate_poetry():
    global word_2_index, index_2_word, wordidx_2_onehot, corpus_num
    w_index = int(np.random.randint(0, corpus_num, 1))
    poetry_list = [index_2_word[w_index]]

    a_prev = np.zeros((1, hidden_num))

    for i in range(23):
        x_onehot = wordidx_2_onehot[w_index]
        h1 = x_onehot @ U + bias_u
        h2 = a_prev @ W + bias_w
        h = h1 + h2
        th = tanh(h)
        pre = th @ V + bias_v

        w_index = int(np.argmax(pre,axis = 1))
        poetry_list.append(index_2_word[w_index])

        a_prev = th
    print("".join(poetry_list))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return 2 * sigmoid(2*x) - 1

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1, keepdims = True)

if __name__ == '__main__':
    datas, word_2_index, index_2_word, wordidx_2_onehot = pro_data("./new_p.txt")

    corpus_num = len(word_2_index)
    hidden_num = 128
    epoch = 1000
    lr = 0.001

    U = np.random.normal(0,2/np.sqrt(corpus_num),size=(corpus_num,hidden_num))
    W = np.random.normal(0,2/np.sqrt(hidden_num),size=(hidden_num,hidden_num))
    V = np.random.normal(0,2/np.sqrt(hidden_num),size=(hidden_num,corpus_num))

    bias_u = np.zeros((1, U.shape[-1]))
    bias_w = np.zeros((1, W.shape[-1]))
    bias_v = np.zeros((1, V.shape[-1]))

    for e in range(epoch):

        for poetry in datas:

            x_onehots,y_onehots = make_x_y(poetry)

            a_prev = np.zeros((1, hidden_num))

            caches = []

            for x_onehot,y_onehot in zip(x_onehots,y_onehots):

                x_onehot = x_onehot[None]
                y_onehot = y_onehot[None]

                h1 = x_onehot @ U + bias_u

                h2 = a_prev @ W + bias_w

                h = h1 + h2

                th = tanh(h)

                pre = th @ V + bias_v

                pro = softmax(pre)

                loss = -np.sum(y_onehot * np.log(pro))

                caches.append((x_onehot, y_onehot, pro, th, a_prev))

                a_prev = th

            dth = 0
            d_W = 0
            d_U = 0
            d_V = 0

            for x_onehot, y_onehot, pro, th, a_prev in reversed(caches):
                G = pro - y_onehot

                delta_V = th.T @ G

                delta_th = G @ V.T + dth

                delta_h = delta_th * (1-th**2)

                delta_W = a_prev.T @ delta_h

                delta_U = x_onehot.T @ delta_h

                dth = delta_h @ W.T

                d_W += delta_W
                d_V += delta_V
                d_U += delta_U

            # ----------- bias 的梯度,等于对应矩阵的行上求和 ------------
            delta_W_bias = np.sum(d_W, axis=0,keepdims = True)
            delta_U_bias = np.sum(d_U, axis=0,keepdims = True)
            delta_V_bias = np.sum(d_V, axis=0,keepdims = True)

            # --------------------- 梯度更新 ------------------------
            W -= lr * d_W
            V -= lr * d_V
            U -= lr * d_U

            bias_w -= lr * delta_W_bias
            bias_u -= lr * delta_U_bias
            bias_v -= lr * delta_V_bias

        generate_poetry()