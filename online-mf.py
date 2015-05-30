import csv
import numpy
import math

D = 2 ** 18

def hash_trick(x):
	return abs(hash(x)) % D

def mean_rate(file_name):
	ex_count = 0
	f = open(file_name)
	reader = csv.reader(f,  delimiter='\t')
	r_sum = 0
	for i, row in enumerate(reader):
		u, i, r = hash_trick(int(row[0])), hash_trick(int(row[1])), int(row[2])
		r_sum += r
		ex_count += 1 	
	mu = r_sum/float(ex_count)
	f.close()
	
	return mu

def matrix_factorization(file_name, K = 10, passes=20, gamma=0.015, lambda0=0.1):

	P = numpy.float64(numpy.random.rand(D, K))
	Q = numpy.float64(numpy.random.rand(K, D))
	BU = [0]*D
	BI = [0]*D

	mu = mean_rate(file_name)
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

def validate(file_name, mu, BU, BI, P, Q):
	loss = 0.
	ex_count = 0.
	f = open(file_name)
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


if __name__ == "__main__":
	numpy.seterr(all='raise')
	mu, BU, BI, P, Q = matrix_factorization('ua.base')
	validate('ua.test', mu, BU, BI, P, Q)	
	validate('ub.test', mu, BU, BI, P, Q)	

# K = 10, passes=20, gamma=0.015, lambda0=0.1, ua.base, ua.test
#Total Loss (SE): 8497.244551, mean loss (MSE): 0.901086, RMSE: 0.949256


