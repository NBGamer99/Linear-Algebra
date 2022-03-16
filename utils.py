import numpy as np
import sys
 
I = np.identity(4)
# w = list()

#Takes a np.Matrix and prints a prettier presentation of the Matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    print("-"*100)
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
    print("-"*100)

    
#Our main function of Guass methode

def GaussSeidel(A,b):
    w = list()
    Aug_A = np.column_stack((A,b))
    n = np.shape(A)[0]
    x = np.array([np.zeros(n)],float).T
    
    for i in range(n):
        if Aug_A[i][i] == 0.0:
            sys.exit('Divide by zero detected!')
   
        for j in range(i+1, n):
            ratio = Aug_A[j][i]/Aug_A[i][i]
            
            for k in range(n+1):
                Aug_A[j][k] = Aug_A[j][k] - ratio * Aug_A[i][k]
            print(f"l{j} -> l{j} - {ratio} * l{i}")
            matprint(Aug_A)
            L = np.identity(4)
            L[j][i] = - ratio
            w.append(L) 
            
    x[n-1] = Aug_A[n-1][n]/Aug_A[n-1][n-1]
    
    for i in range(n-2,-1,-1):
        x[i] = Aug_A[i][n]
     
        for j in range(i+1,n):
            x[i] = x[i] - Aug_A[i][j]*x[j]

        x[i] = x[i]/Aug_A[i][i]

    L = np.linalg.multi_dot([w[5],w[4],w[3],w[2],w[1],w[0]])
    L = np.linalg.inv(L)
    U = np.linalg.multi_dot([w[5],w[4],w[3],w[2],w[1],w[0],A])
    return x,w,L,U


   
def cholesky(A):
    L = np.zeros_like(A, float)
    n = len(L)
    for i in range(n):
        for j in range(i+1):
            if i==j:
                val = A[i,i] - np.sum(np.square(L[i,:i]))
                # if diagonal values are negative return zero - not throw exception
                if val<0:
                    return 0.0
                L[i,i] = np.sqrt(val)
            else:
                L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j]))/L[j,j]
                
    return L
   
   
   
def solver(L,U,b):
  L=np.array(L, float)
  U=np.array(U, float)
  b=np.array(b, float)

  n,_=np.shape(L)
  y=np.zeros(n)
  x=np.zeros(n)

# Forward substitution
  for i in range(n):
    sumj=0
    for j in range(i):
      sumj += L[i,j]*y[j]
    y[i]=(b[i]-sumj)/L[i,i]

# Backward substitution  
  for i in range(n-1, -1, -1):
    sumj=0
    for j in range(i+1,n):
      sumj += U[i,j] * x[j]
    x[i]=(y[i]-sumj)/U[i,i]
  
  return np.array([x])
