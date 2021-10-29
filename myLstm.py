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

    return datas,word_2_index,index_2_word,np.eye(len(word_2_index)).reshape(len(word_2_index),1,-1)

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
    c_prev = np.zeros((1, hidden_num))

    for i in range(31):
        x = wordidx_2_onehot[w_index]
        x_a = np.hstack((x, a_prev))
        f_ = x_a @ F + bf
        i_ = x_a @ I + bi
        c_ = x_a @ C + bc
        o_ = x_a @ O + bo

        ft = sigmoid(f_)
        it = sigmoid(i_)
        cct = tanh(c_)
        ot = sigmoid(o_)

        c_next = ft * c_prev + it * cct
        th = tanh(c_next)
        a_next = ot * th
        pre = a_next @ Y + by

        w_index = int(np.argmax(pre,axis = 1))
        poetry_list.append(index_2_word[w_index])

        a_prev = a_next
        c_prev = c_next
    return "".join(poetry_list)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return 2 * sigmoid(2*x) - 1

def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex,axis = 1, keepdims = True)

if __name__ == '__main__':
    datas, word_2_index, index_2_word, wordidx_2_onehot = pro_data(r"./new_p_7.txt")

    corpus_num = len(word_2_index)
    hidden_num = 50
    epoch = 1000
    lr = 0.05
    max_g = 1

    contact_num = corpus_num + hidden_num

    F = np.random.normal(0, 0.1, size=(contact_num, hidden_num))
    I = np.random.normal(0, 0.1, size=(contact_num, hidden_num))
    C = np.random.normal(0, 0.1, size=(contact_num, hidden_num))
    O = np.random.normal(0, 0.1, size=(contact_num, hidden_num))
    Y = np.random.normal(0, 0.1, size=(hidden_num, corpus_num))

    bf = np.zeros((1, hidden_num))
    bi = np.zeros((1, hidden_num))
    bc = np.zeros((1, hidden_num))
    bo = np.zeros((1, hidden_num))
    by = np.zeros((1, corpus_num))

    for e in range(epoch):
        for p_index,poetry in enumerate(datas):
            a_prev = np.zeros((1,hidden_num))
            c_prev = np.zeros((1,hidden_num))
            x_onehots,y_onehots = make_x_y(poetry)

            caches = []
            for x,y in zip(x_onehots,y_onehots):
                x_a = np.hstack((x,a_prev))
                f_ = x_a @ F + bf
                i_ = x_a @ I + bi
                c_ = x_a @ C + bc
                o_ = x_a @ O + bo

                ft = sigmoid(f_)
                it = sigmoid(i_)
                cct = tanh(c_)
                ot = sigmoid(o_)

                c_next = ft * c_prev + it * cct
                th = tanh(c_next)
                a_next = ot * th
                pre = a_next @ Y + by
                pro = softmax(pre)

                loss = -np.sum(y * np.log(pro))

                caches.append([x_a,ft,c_prev,cct,it,ot,th,a_next,pro,y])

                a_prev = a_next
                c_prev = c_next

            #-------backward--------
            dan = 0
            dcn = 0
            for x_a,ft,c_prev,cct,it,ot,th,a_next,pro,y in reversed(caches):
                G = pro - y
                dY = a_next.T @ G
                d_a_next = G @ Y.T + dan
                dot = th * d_a_next
                do_ = ot * (1-ot) * dot
                dth = ot * d_a_next
                d_c_next = dth * (1-th**2) + dcn
                dcct = it * d_c_next
                dc_ = dcct * (1-cct**2)
                dit = cct * d_c_next
                di_ = dit * it * (1-it)
                dft = c_prev * d_c_next
                df_ = dft * ft * (1-ft)
                dF = x_a.T @ df_
                dI = x_a.T @ di_
                dC = x_a.T @ dc_
                dO = x_a.T @ do_
                dcn = ft * d_c_next
                dan = (df_ @ F.T + di_ @ I.T + dc_ @ C.T + do_ @ O.T)[:,-hidden_num:]

                dF = np.clip(dF, -max_g, max_g)
                dI = np.clip(dI, -max_g, max_g)
                dC = np.clip(dC, -max_g, max_g)
                dO = np.clip(dO, -max_g, max_g)
                dY = np.clip(dY, -max_g, max_g)

                F -= lr * dF
                I -= lr * dI
                C -= lr * dC
                O -= lr * dO
                Y -= lr * dY

                bf -= lr * np.sum(dF, axis=0, keepdims=True)
                bi -= lr * np.sum(dI, axis=0, keepdims=True)
                bc -= lr * np.sum(dC, axis=0, keepdims=True)
                bo -= lr * np.sum(dO, axis=0, keepdims=True)
                by -= lr * np.sum(dY, axis=0, keepdims=True)
            if p_index % 100 == 0:
                print(generate_poetry())