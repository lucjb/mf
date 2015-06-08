import csv
import numpy
import math
import random
import argparse


def hash_trick(x, D):
    return abs(hash(x)) % D


def mean_rate(file_name):
    users = set()
    items = set()
    ex_count = 0
    f = open(file_name)
    reader = csv.reader(f, delimiter='\t')
    r_sum = 0
    for row in reader:
        user = row[0]
        item = row[1]
        users.add(user)
        items.add(item)
        r = float(row[2])
        r_sum += r
        ex_count += 1
    mu = r_sum / float(ex_count)
    f.close()

    return mu, users, items


def matrix_factorization(file_name, K, passes, gamma, lambda0, B, alpha):
    D = 2 ** B
    P = numpy.float64(numpy.random.rand(D, K))
    Q = numpy.float64(numpy.random.rand(K, D))
    BU = [0.] * D
    BI = [0.] * D

    mu, users, items = mean_rate(file_name)

    for i in range(0, D):
        BU[i] = random.random()
        BI[i] = random.random()


    prev_delta_BU = [0.] * D
    prev_delta_BI = [0.] * D
    prev_delta_P = numpy.zeros((D,K))
    prev_delta_Q = numpy.zeros((K,D))

    loss = 0
    ex_count = 0
    for pass_n in range(0, passes):
        f = open(file_name)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            u, i, r = hash_trick(int(row[0]), D), hash_trick(int(row[1]), D), float(row[2])

            eui = r - predict(mu, BU[u], BI[i], P[u, :], Q[:, i])

            ex_count += 1
            loss += eui ** 2

            grad_BU_u = -(eui - lambda0 * BU[u])
            grad_BI_i = -(eui - lambda0 * BI[i])

            delta_BU_u = -(1-alpha)*gamma*grad_BU_u + alpha*prev_delta_BU[u]
            delta_BI_i = -(1-alpha)*gamma*grad_BI_i + alpha*prev_delta_BI[i]

            BU[u] += delta_BU_u
            BI[i] += delta_BI_i

            prev_delta_BU[u] = delta_BU_u
            prev_delta_BI[i] = delta_BI_i

            puo = P[u, :]
            qio = Q[:, i]
            grad_P_u = -(eui * qio - lambda0 * puo)
            grad_Q_i = -(eui * puo - lambda0 * qio)

            delta_P_u = -(1-alpha)*gamma*grad_P_u + alpha*prev_delta_P[u,:]
            delta_Q_i = -(1-alpha)*gamma*grad_Q_i + alpha*prev_delta_Q[:,i]

            P[u, :] += delta_P_u
            Q[:, i] += delta_Q_i

            prev_delta_P[u,:] = delta_P_u
            prev_delta_Q[:,i] = delta_Q_i


            if ex_count % 10000 == 0:
                print loss / float(ex_count), eui**2


        f.close()
    return mu, BU, BI, P, Q, users, items

def predict(mu, BUi, BIj, Pi, Qj):
    pred = 0
    baseline = mu + BUi + BIj
    pred += baseline
    pred += numpy.dot(Pi, Qj)
    return pred

def validate(file_name, mu, BU, BI, P, Q, B):
    D = 2 ** B
    loss = 0.
    ex_count = 0.
    f = open(file_name)
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        u, i, r = hash_trick(int(row[0]), D), hash_trick(int(row[1]), D), float(row[2])
        eui = r - predict(mu, BU[u], BI[i], P[u, :], Q[:, i])
        loss += eui ** 2
        ex_count += 1
    f.close()
    MSE = loss / float(ex_count)
    RMSE = math.sqrt(MSE)
    print 'Total Loss (SE): %f, mean loss (MSE): %f, RMSE: %f' % (loss, MSE, RMSE)


def persist_model(model_file_name, B, K, mu, BU, BI, P, Q, users, items):
        D = 2 ** B
        f = open(model_file_name, 'w')
        f.write("B, %d\n" % B)
        f.write("K, %d\n" % K)
        f.write("mu, %f\n" % mu)

        for user in users:
            u = hash_trick(user, D)
            line_u = "%s, %f, " % (user, BU[u])
            for w in P[u]:
                line_u += "%f, " % w
            line_u = line_u[:-2] + "\n"
            f.write(line_u)

        for item in items:
            i = hash_trick(item, D)
            line_i = "%s, %f, " % (item, BI[i])
            for w in Q[:, i]:
                line_i += "%f, " % w
            line_i = line_i[:-2] + "\n"
            f.write(line_i)

        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', help='Training file. CSV: user, item, score', required=True, dest='input_file')
    parser.add_argument('-k', help='latet variables, default 10', default=10, dest='K', type=int)
    parser.add_argument('-b', help='Number of bits for hash trick, default 18', default=18, dest='b', type=int)
    parser.add_argument('-gamma', help='Learning rate, default 0.015', default=0.015, dest='gamma', type=float)
    parser.add_argument('-l2', help='L2 regularization', default=0, dest='lambda0', type=float)
    parser.add_argument('-passes', help='Input file passes, default 1', default=1, dest='passes', type=int)
    parser.add_argument('-model', help='Model file name', dest='model_file_name')
    parser.add_argument('-predict', help='Prediction file. CSV: user, item.', dest='pred_file_name')
    parser.add_argument('-alpha', help='Momentum weigh. [0,1]. Defualt: 0.5', default=0.5, dest='alpha', type=float)

    args = parser.parse_args()
    input_file, K, passes, gamma, lambda0, B, model_file_name, pred_file_name, alpha= args.input_file, args.K, args.passes, args.gamma, args.lambda0, args.b, args.model_file_name, args.pred_file_name, args.alpha

    numpy.seterr(all='raise')

    mu, BU, BI, P, Q, users, items = matrix_factorization(input_file, K, passes, gamma, lambda0, B, alpha)

    if model_file_name:
        persist_model(model_file_name, B, K, mu, BU, BI, P, Q, users, items)

    print 'Training error:'
    validate(input_file, mu, BU, BI, P, Q, B)

    if pred_file_name:
        print pred_file_name + ' evaluation:'
        validate(pred_file_name, mu, BU, BI, P, Q, B)

'''
K = 10
passes=20
gamma=0.015
lambda0=0.1
B=18

ua.base Total Loss (SE): 60711.228088, mean loss (MSE): 0.670324, RMSE: 0.818733
Held out set error:
ua.test Total Loss (SE): 8486.100675, mean loss (MSE): 0.899905, RMSE: 0.948633
ub.test Total Loss (SE): 6243.402562, mean loss (MSE): 0.662079, RMSE: 0.813682

''' 

#python online-mf.py -d /home/lbernardi/code/lbernardi/destRecom/mf-train.tsv -passes 1 -k 10 -predict /home/lbernardi/code/lbernardi/destRecom/mf-test.tsv -l2 1 -gamma 0.2