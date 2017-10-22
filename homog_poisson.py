import random
import math
import numpy as np
from collections import defaultdict



def h_poi(count,lmbda,t,i):
                arrival = []
                inter = []
		while True:
			u = random.uniform(0,1)
			w = -(1.0/lmbda)*(math.log(1-u))
			t = t+w
			if i>=count:
				break
			else:
				i = i+1
                                inter.append(w)
                                arrival.append(t)
		
                return inter
		

def gen_data(c,l,ts,idx):
                d_count = c
		lmbda = l
		t = ts
		i = idx
		arr = h_poi(d_count,lmbda,t,i)	
                return arr   


def peace():
        f = gen_data(10000,0.21,0,0)
        print np.mean(f),0.21,1.0/0.21



def main():
	trials = 20000
	pt = []
	for y in range(1,11,1):
                        dt = defaultdict(list)
			for x in range(trials):
					arrivals = gen_data(100,0.1,0,0)
				        #arrivals.insert(0,0)
				        lor = 1.0/0.1
					start_p = arrivals[-(y+1)]
					data_p = arrivals[-y:]
					predict_p = []
					for x1 in data_p:
						start_p = start_p+lor
						predict_p.append(start_p)
					temp = np.sqrt(np.mean(np.square(np.array(data_p) - np.array(predict_p))))
				        #print arrivals[-10:]
					pt.append(temp)
					arr = arrivals[:-y]
				        arr1 = np.array(arr)[:1] - np.array(arr)[:-1]
				        #print type(arr1),len(arr1)
					arr_r = arr[::-1]
					for i in range(1,len(arr_r),1):
							tar = arr_r[:i]
							lmbda = np.mean(tar)
							start_p = arrivals[-(y+1)]
							data_p = arrivals[-y:]
							predict_p = []
							for x1 in data_p:
									start_p = start_p+lmbda
									predict_p.append(start_p)
							temp = np.sqrt(np.mean(np.square(np.array(data_p) - np.array(predict_p))))                
							dt[i].append(temp)
			
			f = open("poisson_test_"+str(y)+".txt","w")	
			for x in dt:
				f.write(str(x)+"\t"+str(np.mean(pt))+"\t"+str(np.mean(dt[x])))
				f.write("\n")
			#print "***********************************************************"
			f.close()
				        
main()
