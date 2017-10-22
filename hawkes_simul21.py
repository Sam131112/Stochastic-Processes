import numpy as np
import random as rn
import pickle
#import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import codecs
from collections import OrderedDict
from collections import defaultdict
import scipy.stats as st
import random
import networkx as nx
import time
import sys
import difflib




def loglik(tm,param):
		t = np.array(tm)
		R = [0.0 for x in xrange(len(tm))]
		for i in range(len(tm)):
					        if i==0:
					                R[i] = 0.0
				        	else:
				                	k = math.exp(-param[2]*(t[i] - t[i-1]))*(1+R[i-1])
				                	R[i]=k
	    	r = np.sum(np.log(param[1]*np.array(R)+param[0]))
		return (param[0]*tm[-1]-(param[1]/param[2])*np.sum(np.exp(-param[2]*(t[-1]-t))-1)-r)


def loglik2(tm,param):
		t = np.array(tm)
		R = [0.0 for x in xrange(len(tm))]
		for i in range(len(t)):
					        if i==0:
					                R[i] = 0.0
				        	else:
							tm1 = t[i] - t[:i]
				                	k = np.sum(np.exp(-param[2]*tm1))
				                	R[i]=k
	    	r = np.sum(np.log(param[1]*np.array(R)+param[0]))
		return (param[0]*tm[-1]-(param[1]/param[2])*np.sum(np.exp(-param[2]*(t[-1]-t))-1)-r)




def loglik_der(tmp,param):
		t = np.array(tmp)
		B = [0.0 for x in range(len(t))]
		for i in range(len(t)):
					if i == 0:
						B[i] = 0
					else:
						tm = t[i] - t[:i]
						tm1 = np.exp(-param[2]*tm)
						B[i] = np.dot(tm,tm1)
		A = [0.0 for x in xrange(len(t))]
		for i in range(len(t)):
					        if i==0:
					                A[i] = 0.0
				        	else:
				                	k = math.exp(-param[2]*(t[i] - t[i-1]))*(1+A[i-1])
				                	A[i]=k
		
		A_n = np.array(A)
		A_d = A_n*param[1]+param[0]
		A_f = A_n/A_d
		dl_alpha = -math.pow(param[2],-1)*np.sum(np.exp(-param[2]*(t[-1]-t))-1)-np.sum(A_f)
		dl_mu = t[-1] - np.sum(1.0/A_d)
		B_n = param[1]*np.array(B)
		B_f = B_n/A_d
		term1  = (t[-1]-t)*np.exp(-param[2]*(t[-1]-t))*math.pow(param[2],-1)
		term2  = np.exp(-param[2]*(t[-1]-t))*math.pow(param[2],-2)
		t_f = (term1+term2)*param[1]
		dl_beta = np.sum(t_f)+np.sum(B_f)
		return np.array([dl_mu,dl_alpha,dl_beta])
		
	


def loglik_hess(tmp,param):
		t = np.array(tmp)
		B = [0.0 for x in range(len(t))]
		for i in range(len(t)):
					if i == 0:
						B[i] = 0.0
					else:
						tm = t[i] - t[:i]
						tm1 = np.exp(-param[2]*tm)
						B[i] = np.dot(tm,tm1)
		A = [0.0 for x in xrange(len(t))]
		for i in range(len(t)):
					        if i==0:
					                A[i] = 0.0
				        	else:
				                	k = math.exp(-param[2]*(t[i] - t[i-1]))*(1+A[i-1])
				                	A[i]=k
		C = [0.0 for x in xrange(len(t))]
		for i in range(len(t)):
					if i == 0:
						C[i] = 0.0
					else:
						tm = t[i] - t[:i]
						tm1 = np.exp(-param[2]*tm)
						C[i] = np.dot(tm**2,tm1)
		A_n = np.array(A)
		B_n = np.array(B)
		A_d = A_n*param[1]+param[0]
		n_1 = param[1]*A_n*B_n
		A_d_2 = A_d**2
		term7 =  (np.exp(-param[2]*(t[-1]-t))-1)*math.pow(param[2],-2)
		term1  = (t[-1]-t)*np.exp(-param[2]*(t[-1]-t))*math.pow(param[2],-1)
		term2  = ((t[-1]-t)**2)*np.exp(-param[2]*(t[-1]-t))*math.pow(param[2],-1)
		term3  = (t[-1]-t)(np.exp(-param[2]*(t[-1]-t))-1)*math.pow(param[2],-2)
		term4  = (np.exp(-param[2]*(t[-1]-t))-1)*math.pow(param[2],-3)
		C_n = param[1]*np.array(C)
		term5 = C_n/A_d
		term6 = ((param[1]*B_n)/A_d)**2
		term8 = param[1]*A_n*B_n
		dmu_al = np.sum(A_n/(A_d)**2)
		dmu_bi = -np.sum((param[1]*B_n)/(A_d)**2)
		dal_2 = np.sum((A_n/A_d)**2)
		dbi_2 = -param[1]*np.sum(term2+term3*2+term4*2) - np.sum(term5-term6)
		dmu_2 = np.sum(1.0/(A_d)**2)
		dbi_al = np.sum(term7+term1) - np.sum(-(B_n/A_d)+term8/(A_d)**2)
		return np.array([[dmu_2,dmu_al,dmu_bi],[dmu_al,dal_2,dbi_al],[dmu_bi,dbi_al,dbi_2]])			
	
	


def lambda_func(times,s,a,b,c):
	if s == 0:
		return a
	elif len(times) == 0:
		return a
	else:							
		l = a + np.sum(b*np.exp(-c*(s-np.array(times))))
		return l



def hawks_estimation(x,event):
		h_est = lambda x:loglik(event,x) 
                h_jac = lambda x:loglik_der(event,x)
		h_hess = lambda x:loglik_hess(event,x)
		cons = ({'type': 'ineq','fun' : lambda x: np.array([x[2] - x[1]-1e-5])})
		d =  minimize(h_est,x,method="SLSQP",bounds=((1e-5,None),(1e-5,None),(1e-5,None)),constraints=cons,options={'disp':False,'maxiter':500}) 
		return d


## Data Generated

def main():
          
		#d_sizes = [1,2,4,6,8,10]
		d_sizes = ["all"]
		counts = 200
		rounds = 1000
		original = []
		check = [1,2,3,4,5,6,7,8,9,10]
                for ch in [1]:		
			estimates = defaultdict(list)
			for sz1 in range(rounds):
                                                   
							mu = 0.009
							al = 2.75
							bi = 3.89
							times = []
							s = 0
							lmdb = 0
							i = 0
							while i<counts:
								lmdb = lambda_func(times,s,mu,al,bi)
								u = rn.uniform(0,1)
								w = -np.log(u)/lmdb
								s=s+w
								D = rn.uniform(0,1)
								if D*lmdb <= lambda_func(times,s,mu,al,bi):
									times.append(s)
									i = i + 1
							
							n_times = times[:-ch]
							target = times[-ch:]
							for xt in target:	
								lambda_predict = lambda_func(n_times,n_times[-1],0.009,2.75,3.89)
								mean_predict = 1.0/lambda_predict
								n_times.append(mean_predict+n_times[-1])
                                                                #print lambda_predict,mean_predict,n_times[-1],target[-1],len(times),len(n_times)
							
                                                        
							predicts = n_times[-ch:]
							k = np.sum(np.square(np.array(predicts) - np.array(times[-ch:])))
							k = (1.0*k)/len(target)
							k = np.sqrt(k)
							original.append(k)
							last = len(times)
                                                        print sz1
							for  indexes  in range(20,last,1):
									n_times_k = times[-indexes:-ch]
									maxs = 1e10
									est = []
									tm1 = n_times_k[:]		
									#print "Train ",tm1
									for x1 in range(5):                        
										a = np.random.uniform(0,1)
										b = np.random.uniform(0,1)
										c = np.random.uniform(1.1,2)
										pr = hawks_estimation([a,b,c],tm1)
										if pr.fun<maxs:
												maxs = pr.fun
												est = pr.x
									
									mu1 = est[0]
									al1 = est[1]
									bi1 = round(est[2],2)
                                                                        #print mu1,al1,bi1
									n_times = times[:-ch]
									target = times[-ch:]
									for xt in target:	
										lambda_predict = lambda_func(n_times_k,n_times_k[-1],mu1,al1,bi1)
										mean_predict = 1.0/lambda_predict
										n_times_k.append(mean_predict+n_times_k[-1])
                                                                        predicts = n_times_k[-ch:]
									k = np.sum(np.square(np.array(predicts) - np.array(target)))
									k = (1.0*k)/len(target)
									k = np.sqrt(k)
                                                                        #print indexes,mu1,al1,bi1,k
									estimates[indexes].append(k)
		        fls = open("MMy_hawks_"+str(ch)+".txt","w")
			for x in estimates:
                                        fls.write(str(x)+"\t"+str(np.mean(original))+"\t"+str(np.mean(estimates[x])))
                                        fls.write("\n")
					#print x,np.mean(original),np.mean(estimates[x])		
                        fls.close()


main()

#print mu,al,bi


