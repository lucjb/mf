import csv
import numpy
import math

D = 2 ** 20

def hash_trick(x):
	return abs(hash(x)) % D

def matrix_factorization(file_name, K = 32, passes=50, gamma=0.015, lambda0=0.1):

	P = numpy.float64(numpy.random.rand(D, K))
	Q = numpy.float64(numpy.random.rand(K, D))
	BU = [0]*D
	BI = [0]*D

	ex_count = 0
	f = open(file_name)
	reader = csv.reader(f,  delimiter='\t')
	r_sum = 0
	for i, row in enumerate(reader):
		u, i, r = hash_trick(int(row[0])), hash_trick(int(row[1])), int(row[2])
		BU[u] += r/float(D)
		BI[i] += r/float(D)
		r_sum += r
		ex_count += 1 	
	mu = r_sum/float(ex_count)
	f.close()

	print 'Priors: %d' % mu
	
	loss = 0
	ex_count = 0
	for pass_n in range(0, passes):
		f = open(file_name)
		reader = csv.reader(f,  delimiter='\t')
		for i, row in enumerate(reader):
			u, i, r = hash_trick(int(row[0])), hash_trick(int(row[1])), int(row[2])
			eui = r - predict(mu, BU[u], BI[i], P[u,:], Q[:,i])
			ex_count+=1
			loss += eui**2

			BU[u]+=gamma*(eui-lambda0*BU[u])
			BI[i]+=gamma*(eui-lambda0*BI[i])	

			puo = P[u,:]
			qio = Q[:,i]

			P[u,:]+=gamma*(eui*qio - lambda0*puo)
			Q[:,i]+=gamma*(eui*puo - lambda0*qio)

			if ex_count % 10000==0:
				print loss/float(ex_count)
		f.close()
	return mu, BU, BI, P, Q

def predict(mu, BUi, BIj, Pi, Qj):
	pred = 0
	baseline = mu + BUi + BIj
	pred += baseline
	pred += numpy.dot(Pi, Qj)
	return pred



if __name__ == "__main__":
	numpy.seterr(all='raise') 
	
	mu, BU, BI, P, Q = matrix_factorization('ua.base')
	
	loss = 0.
	ex_count = 0.
	f = open('ua.test')
	reader = csv.reader(f,  delimiter='\t')
	r_sum = 0
	for i, row in enumerate(reader):
		u, i, r = hash_trick(int(row[0])), hash_trick(int(row[1])), int(row[2])
		eui = r - predict(mu, BU[u], BI[i], P[u,:], Q[:,i])
		loss += eui**2
		ex_count += 1		
	f.close()
	MSE = loss/float(ex_count)
	RMSE = math.sqrt(MSE)
	print 'Total Loss (SE): %f, mean loss (MSE): %f, RMSE: %f' %(loss, MSE, RMSE)

# steps=50, alpha=0.015, beta0=0.0001, beta1=0.1, k=32
# 0.687426219807
# 0.94677112744

