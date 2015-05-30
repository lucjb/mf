import csv
import numpy

def matrix_factorization(R, mu, BU, BI, P, Q, K, steps=40, gamma=0.015, lambda0=0.1):
    Q = Q.T
    prev_cost = 100000000
    ex_count = 0
    loss = 0
    for step in xrange(steps):
        for u in xrange(len(R)):
            for i in xrange(len(R[u])):
                if R[u,i] > 0:
		    ex_count+=1
                    eui = R[u,i] - predict(mu, BU[u], BI[i], P[u,:], Q[:,i])
		    loss += eui**2

 		    BU[u]+=gamma*(eui-lambda0*BU[u])
		    BI[i]+=gamma*(eui-lambda0*BI[i])	

		    puo = P[u,:]
		    qio = Q[:,i]

                    P[u,:]+=gamma*(eui*qio - lambda0*puo)
    		    Q[:,i]+=gamma*(eui*puo - lambda0*qio)
	
#	cost = e(R, mu, BU, BI, P, Q, K, lambda0)
#	improvement = prev_cost - cost
#	print step, prev_cost, cost, improvement
	print step, loss/float(ex_count)
#	prev_cost = cost
#	if improvement <0:
#           break


def predict(mu, BUi, BIj, Pi, Qj):
	pred = 0
	baseline = mu + BUi + BIj
	pred += baseline
	pred += numpy.dot(Pi, Qj)
	return pred


def e(R, mu, BU, BI, P, Q, K, lambda0):
	e = 0
	for u in xrange(len(R)):
	    for i in xrange(len(R[u])):
		if R[u,i] > 0:
		    e += (R[u,i] - predict(mu, BU[u], BI[i], P[u,:], Q[:,i]))**2
		    e += lambda0 * (numpy.sum(P[u,:]**2 + Q[:,i]**2) + BU[u]**2 + BI[i]**2)
	return e

def load(file_name):
    R = numpy.zeros((943, 1682))
    reader = csv.reader(open(file_name),  delimiter='\t')
    for i, row in enumerate(reader):
	u, i, r = int(row[0]), int(row[1]), int(row[2])
	R[u-1,i-1]=r
    return R

def rmse(R, mu, BU, BI, P, Q):
    e = 0
    t = 0
    for i in xrange(len(R)):
	for j in xrange(len(R[i])):
		if R[i, j] > 0:
			t += 1
			e += (R[i, j] - predict(mu, BU[i], BI[j], P[i,:], Q[:,j]))**2
    e/=t
    e = numpy.sqrt(e)	
    return e

if __name__ == "__main__":
    numpy.seterr(all='raise') 
    R = load('ua.base')
        
    N = len(R)
    M = len(R[0])
	
    mu = numpy.mean(R)
    BU = mu - numpy.mean(R, axis=1)
    BI = mu - numpy.mean(R, axis=0)

    K = 32
    P = numpy.float64(numpy.random.rand(N,K))
    Q = numpy.float64(numpy.random.rand(M,K))

    matrix_factorization(R, mu, BU, BI, P, Q, K)

    print rmse(R, mu, BU, BI, P, Q.T)
    T = load('ua.test')
    print rmse(T, mu, BU, BI, P, Q.T)

# steps=50, alpha=0.015, beta0=0.0001, beta1=0.1, k=32
# 0.687426219807
# 0.94677112744

