import csv
import numpy
import math
import random

def hash_trick(x, D):
	return abs(hash(x)) % D 

def mean_rate(file_name):
	users = set()	
	items = set()
	ex_count = 0
	f = open(file_name)
	reader = csv.reader(f,  delimiter='\t')
	r_sum = 0
	for row in reader:
		user = row[0]
		item = row[1]
		users.add(user)
		items.add(item)
		r = int(row[2])
		r_sum += r
		ex_count += 1 	
	mu = r_sum/float(ex_count)
	f.close()
	
	return mu, users,items

def matrix_factorization(file_name, K = 10, passes=1, gamma=0.015, lambda0=0.1, B=18):
	D = 2 ** B
	P = numpy.float64(numpy.random.rand(D, K))
	Q = numpy.float64(numpy.random.rand(K, D))
	BU = [0.]*D
	BI = [0.]*D
	for i in range(0, D):
		BU[i] = random.random()
		BI[i] = random.random()
	mu, users, items = mean_rate(file_name)

	loss = 0
	ex_count = 0
	for pass_n in range(0, passes):
		f = open(file_name)
		reader = csv.reader(f,  delimiter='\t')
		for row in reader:
			u, i, r = hash_trick(int(row[0]), D), hash_trick(int(row[1]), D), int(row[2])

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
	return mu, BU, BI, P, Q, users, items

def predict(mu, BUi, BIj, Pi, Qj):
	pred = 0
	baseline = mu + BUi + BIj
	pred += baseline
	pred += numpy.dot(Pi, Qj)
	return pred

def validate(file_name, mu, BU, BI, P, Q, B):
	D = 2**B
	loss = 0.
	ex_count = 0.
	f = open(file_name)
	reader = csv.reader(f,  delimiter='\t')
	r_sum = 0
	for row in reader:
		u, i, r = hash_trick(int(row[0]), D), hash_trick(int(row[1]), D), int(row[2])
		eui = r - predict(mu, BU[u], BI[i], P[u,:], Q[:,i])
		loss += eui**2
		ex_count += 1		
	f.close()
	MSE = loss/float(ex_count)
	RMSE = math.sqrt(MSE)
	print 'Total Loss (SE): %f, mean loss (MSE): %f, RMSE: %f' %(loss, MSE, RMSE)


if __name__ == "__main__":
	numpy.seterr(all='raise')
	K = 10
	passes=20
	gamma=0.015
	lambda0=0.1
	B=18
	
	mu, BU, BI, P, Q, users, items = matrix_factorization('ua.base', K, passes, gamma, lambda0, B)
	

	D = 2**B
	f = open('mf.model', 'w')
	f.write("B, %d\n" % B)
	f.write("K, %d\n" % K)
	f.write("mu, %f\n" % mu)

	for user in users:
		u = hash_trick(user, D)
		line_u = "%s, %f, " % (user, BU[u])
		for w in P[u]:
			line_u += "%f, " % w			
		line_u = line_u[:-2]+"\n"
		f.write(line_u)

	for item in items:
		i = hash_trick(item, D)
		line_i = "%s, %f, " % (item, BI[i])
		for w in Q[:,i]:
			line_i += "%f, " % w			
		line_i = line_i[:-2]+"\n"
		f.write(line_i)
	
	f.close()
		
	print 'Training error:'
	validate('ua.base', mu, BU, BI, P, Q, B)
	print 'Held out set error:'	
	validate('ua.test', mu, BU, BI, P, Q, B)	
	validate('ub.test', mu, BU, BI, P, Q, B)	

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

