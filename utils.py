import numpy as np
import sys
 
I = np.identity(4)

#Takes a np.Matrix and prints a prettier presentation of the Matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    print("-"*100)
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
    print("-"*100)

def GaussSeidel(A,b):
    # w = list()
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
                # w.append(L[j][i] = - ratio) 
            print(f"l{j} -> l{j} - {ratio} * l{i}")
            matprint(Aug_A)
            
    x[n-1] = Aug_A[n-1][n]/Aug_A[n-1][n-1]
    
    for i in range(n-2,-1,-1):
        x[i] = Aug_A[i][n]
     
        for j in range(i+1,n):
            x[i] = x[i] - Aug_A[i][j]*x[j]

        x[i] = x[i]/Aug_A[i][i]
    
    # U = np.dot(A,L.T)
    return x


    
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


if __name__ == '__main__':
    A = np.array([[4, 4, 2, 0], [4, 5, 0, 0], [2, 0, 6, 1], [0, 0, 1, 2]], dtype='f')
    b = np.array([[1,2,3,4]],dtype='f').T
    Aug_A_b = np.column_stack((A,b))
    print("La matrice A :")
    matprint(A)
    print("La matrice b")
    matprint(b)
    print("La matrice augmente est :")
    matprint(Aug_A_b)
    x1,L1,U1 = GaussSeidel(A,b)
    print("La solution par methode de Gauss")
    matprint(x1.T)
    print("La solution L1 ")
    matprint(L1)
    print("La solution U1")
    matprint(U1)
    print("solution LU :")
    x2 = solver(L1,U1,b)
    matprint(x2)
    
    # L = cholesky(A)
    # matprint(L)
    # U = np.transpose(L)
    # matprint(U)
    # print("La solution par methode de Cholesky")
    # x2 = solver(L, U, b)
    # matprint(x2)
    # print(x2)
    # matprint(inv_w)
    # print("Le produit de A*b")
    # prd = np.dot(A,b)
    # matprint(prd)
    # print("-"*100,"\nLa matrice lower A :\n","-"*100)
    # L = np.tril(A)
    # matprint(L)
    # print("-"*100,"\nLa matrice upper A :\n","-"*100)
    # U = np.triu(A)
    # matprint(U)
    # print("-"*100,"\nLa matrice diag A :\n","-"*100)
    # diag = np.diag(np.diag(A))
    # # matprint(diag)


    # print("-"*100,"\nLes sous matrice A :\n","-"*100)
    # a1 = A[0:1,0:1]
    # matprint(a1)
    # print(f"sont det(a1) = {np.linalg.det(a1):0.4}")
    # print("-"*100,"\nLes sous matrice A :\n","-"*100)
    # a2 = A[0:2,0:2]
    # matprint(a2)
    # print(f"sont det(a22) = {np.linalg.det(a2):0.4} ")
    # print("-"*100,"\nLes sous matrice A :\n","-"*100)
    # a3 = A[0:3,0:3]
    # matprint(a3)
    # print(f"sont det(a3) = {np.linalg.det(a3):0.4} ")
    # print("-"*100,"\nLes sous matrice A :\n","-"*100)
    # a4 = A[0:4,0:4]
    # matprint(a4)
    # print(f"sont det(a4) = {np.linalg.det(a4):0.4} ")
    # print("-"*100,"\nLa matrice B\n","-"*100)
    # n = 3
    # b = np.zeros((n,n+1))
    # matprint(b)


