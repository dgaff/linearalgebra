import numpy as np
from numpy.linalg import inv

'''
Standard matrix formula for solving a system of equations. A is an nxn matrix.

Ax = b
Ai A x = Ai b
I x = Ai b
x = Ai b
'''

# Test matrix. An underdetermined example is commented out to test exception handling.
A = np.array([[1.0, 2.0, 3.0],[10.0, 7.0, 3.0],[3.0, 1.0, 11.0]])
# A = np.array([[1.0, 2.0, 3.0],[1.0, 2.0, 3.0],[3.0, 1.0, 11.0]])

# Get the inverse of A.
try:
    Ai = inv(A)
except np.linalg.LinAlgError:
    print("Matrix is singular")
    quit()

print("\nStandard method for solving Ax = b using an inverse")
print("A")
print(A)
print("Matrix A rank: ", np.linalg.matrix_rank(A))
print("Ai")
print(Ai)
print("Ai A = I")
print(np.dot(Ai, A))

# Calculate b from some test x values.
xtest = np.array([10.0, 11.0, 12.0])
b = np.dot(A, xtest)
print("b")
print(b)

# Now solve for x and show that they match.
x = np.dot(Ai, b)
print("xtest and x calculated")
print (xtest, x)

'''
SVD way - this is a least squares fit for use when there is error in the system.
https://en.wikipedia.org/wiki/Singular_value_decomposition
https://austingwalters.com/using-svd-to-obtain-regression-lines/
See PDFs in this repo, too.

For:

Ax = b

A can be decomposed by SVD as follows:

A = U S Vt      

U and V are orthogonal matrices for n x n systems, so transpose == inverse. Transpose == inverse
for these matrices on m x n systems (m > n), too. S is diagonal matrix containing the singular values. 
If A is overdetermined (more rows than columns), S must have zeroed rows added in order to calc x.
The svd() function just returns the vector of values in S as the s vector. 

U S Vt x = b

x = V Si Ut b

Si = is a pseudo-inverse in which the non-zero elements on the diagonal are inverted.
'''

# Reusing A, b, and xtest from above.

# SVD
# Note: catch LinAlgError to see if the calculation doesn't converge. I test this above.
U, s, Vt = np.linalg.svd(A)

# Create a diagonal matrix for S and it's pseudo-inverse
S = np.diag(s)
si = 1/s
Si = np.diag(si)

print("\nSVD method of solving Ax = b")
print("A = U S Vt")
print("U ", U.shape)
print(U)
print("S", S.shape)
print(S)
print("s ", s.shape)
print(s)
print("Si", Si.shape)
print(Si)
print("si ", si.shape)
print(si)
print("Vt ", Vt.shape)
print(Vt)

print("\nNow let's check the decomposition")
print("A")
print(A)

# element-wise multiply works for s because s is a 1D vector (??)
# Acheck = np.dot(U * s, Vt)  
# print("A = U s Vt")
# print(Acheck)

# Check the decomposition.
Acheck2 = np.dot(U, np.dot(S, Vt))
print("A = U S Vt")
print(Acheck2)

print("\nNow let's solve for x")

# Solve for x. This just uses the si vector instad of the Si matrix. 
xnew = np.dot(Vt.T * si, np.dot(U.T, b))
print("x = V si Ut b")
print(xnew)

# Solve for x again using the Si matrix. I prefer this notation better.
# Note: there are lots of ways to write this matrix multiplication. multi_dot is the clearest to me.
# Also note that technically this should be Si.T as below. For a square matrix, it doesn't matter.
# xnew = np.dot(np.dot(Vt.T, Si), np.dot(U.T, b))
# xnew = Vt.T.dot(Si).dot(U.T).dot(b)
xnew = np.linalg.multi_dot([Vt.T, Si, U.T, b])
print("x = V Si Ut b")
print(xnew)
print("\n")

'''
SVD again, but with an overdetermined system of equations, i.e. more rows than columns.

Also adding some error into the system to show an "almost correct" answer.
'''

print("\n SVD again with a non-square matrix B")

# Test matrix
B = np.array([[1.0, 2.0, 3.0],[10.0, 7.0, 3.0],[3.0, 1.0, 11.0],[15.0, 9.0, 5.0]])
print("B")
print(B)
print("Matrix B rank: ", np.linalg.matrix_rank(B))

# Put in some x values to calculate b
xtest2 = np.array([10.0, 11.0, 12.0])
b = np.dot(B, xtest2)
print("xtest")
print(xtest2)
print("b")
print(b)

# Add some error to B to show least-squares convergence.
print("\nLet's add some error to B")
error = np.random.rand(4,3)
error /= 100.0
print(error)
B += error
print("\nB with error")
print(B)

# SVD
U, s, Vt = np.linalg.svd(B)

# Create a diagonal matrix for S and it's pseudo-inverse
S = np.diag(s)
si = 1/s
Si = np.diag(si)

# Addd a row to S since we're over-determined by one row. (TODO: fix to work with any m x n.)
S = np.append(S, [[0, 0, 0]], axis=0)
Si = np.append(Si, [[0, 0, 0]], axis=0)

print("\nSVD method of solving Bx = b")
print("B = U S Vt")
print("U ", U.shape)
print(U)
print("S", S.shape)
print(S)
print("s ", s.shape)
print(s)
print("Si", Si.shape)
print(Si)
print("Si.T", Si.T.shape)
print(Si.T)
print("si ", si.shape)
print(si)
print("Vt ", Vt.shape)
print(Vt)

print("\nNow let's check the decomposition")
print("B")
print(B)

# Check the decomposition
Bcheck2 = np.linalg.multi_dot([U, S, Vt])
print("B = U S Vt")
print(Bcheck2)

# Solve for x
# Note that Si has to be transposed to make it 3x4. The diagonal values don't change on the transpose.
# The zero row just becomes a zero column.
xnew2 = np.linalg.multi_dot([Vt.T, Si.T, U.T, b])
print("x = V Si Ut b")
print(xnew2)
print("\n")
