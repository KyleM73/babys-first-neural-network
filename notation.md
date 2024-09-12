# Notation
- $\in$ means "element of" a set
- $\mathbb{R}$ is the space of real numbers
- $\mathbb{R}^n$ is the space of real-valued vectors of dimension $n$ (implicitly $n\times 1$)
- $\mathbb{R}^{n\times m}$ is the space of real-valued matrices of dimension $n\times m$
- $\forall$ means "for all", e.g. "$0x=0\forall x\in\mathbb{R}$" means "zero times $x$ is zero for any real number $x$".
- Lowercase letters denote a (column) vector, e.g. $x,y\in\mathbb{R}^n$
- Uppercase letters denote a matrix, e.g. $A,B\in\mathbb{R}^{n\times m}$
- The vector transpose $x^T\in\mathbb{R}^n$ swaps between column and row vectors, e.g. $\begin{bmatrix}x_{0}\ x_{1}\end{bmatrix}^T=\begin{bmatrix}x_{0}\\ x_{1}\end{bmatrix}$
- The matrix transpose $A^T\in\mathbb{R}^{m\times n}$ swaps the last two dimensions of a matrix, e.g. $\begin{bmatrix}a_{0,0}\ a_{0,1}\ a_{0,2}\\ a_{1,0}\ a_{1,1}\ a_{1,2}\end{bmatrix}^T=\begin{bmatrix}a_{0,0}\ a_{1,0}\\ a_{0,1}\ a_{1,1}\\ a_{0,2}\ a_{1,2} \end{bmatrix}$
- Vectors and matrices are indexed in (row, column) order
    * $A_{i,j}$ is the element of $A$ in the $i$th row and the $j$th column, $\text{dim}(A_{i,j})=1$
    * $A_i$ is the $i$th row of $A$, $\text{dim}(A_i)=m$
    * The $j$th column of $A$ can be found via $A_j^T$, $\text{dim}(A_j^T)=n$
- $\hat y$ denotes that the quantity $y$ is estimated
- $f:X\rightarrow Y$ defines the function $f$ mapping the set $X$ to the set $Y$, i.e. "$y=f(x),\ x\in X,\ y\in Y$"
- The inner product (also called the dot product) of two vectors is given by $x^Ty=\sum_{i=0}^nx_iy_i\in\mathbb{R}$
- $\nabla$ is the gradient operator
    * Let $f: \mathbb{R}\times \mathbb{R}\rightarrow \mathbb{R},\ f=f(p,q)$, then $\nabla f=\begin{bmatrix}\frac{\partial f}{\partial p}\ \frac{\partial f}{\partial q}\end{bmatrix}^T$

## Matrix and Vector Derivatives
Let $p,q \in\mathbb{R},\ x\in\mathbb{R}^n,\ y\in\mathbb{R}^m,\ A\in\mathbb{R}^{n\times m}$

### Scalar with respect to scalar:
- $\frac{d}{dp}q=\frac{dq}{dp}\in\mathbb{R}$
- $\frac{d}{dp}p=1\in\mathbb{R}$

### Vector with respect to scalar:
- $\frac{dx}{dp}=\begin{bmatrix}\frac{dx_0}{dp}\dots \frac{dx_n}{dp}\end{bmatrix}^T\in\mathbb{R}^n$

### Scalar with respect to vector:
- $\frac{\partial p}{\partial x}=\begin{bmatrix}\frac{\partial p}{\partial x_0}\dots \frac{\partial p}{\partial x_n}\end{bmatrix}\in\mathbb{R}^n$
- Let $f:\mathbb{R}^n\rightarrow\mathbb{R}$, then $\nabla f_x=\frac{\partial f}{\partial x}^T\in\mathbb{R}^n$

### Vector with respect to vector:
- $\frac{\partial y}{\partial x}=\begin{bmatrix}\frac{\partial y_0}{\partial x_0}\dots \frac{\partial y_0}{\partial x_n}\\ \vdots\\ \frac{\partial y_m}{\partial x_0}\dots \frac{\partial y_m}{\partial x_n}\end{bmatrix}\in\mathbb{R}^{m\times n}$
- $\frac{\partial x}{\partial x}=I\in\mathbb{R}^{n\times n}$
- Let $y=A^Tx$, then $\frac{dy}{dx}=A^T\in\mathbb{R}^{m\times n}$

### Vector with respect to matrix:
- $\frac{\partial x}{\partial A}=\begin{bmatrix}\frac{\partial x_0}{\partial A}\\ \vdots\\ \frac{\partial x_n}{\partial A}\end{bmatrix}=\begin{bmatrix}\frac{\partial x}{\partial A_0}\ \dots\ \frac{\partial x}{\partial A_n}\end{bmatrix}\in\mathbb{R}^{n\times n\times m}$

    Note that the result is a tensor of order three. In special cases computing this tensor explicitly may be avoided by taking the derivatives component-wise. Consider the following:

    Let $y=A^Tx,\ z\in\mathbb{R}^m$, then:
        
    $\begin{bmatrix}y_0\\ \vdots\\ y_m\end{bmatrix}=\begin{bmatrix}A_0^Tx\\ \vdots\\ A_m^Tx\end{bmatrix}$

    $\frac{\partial y}{\partial A}=\begin{bmatrix}\frac{\partial y_0}{\partial A_0}\\ \vdots\\ \frac{\partial y_m}{\partial A_m}\end{bmatrix}=\begin{bmatrix}x^T &  &  &   \\&  x^T&  &  \\&  & \ddots  &  \\&  &  & x^T\end{bmatrix}\in\mathbb{R}^{m\times mn}$

    $z\frac{\partial y}{\partial A}=\begin{bmatrix}z_0\\ \vdots\\ z_m\end{bmatrix}\begin{bmatrix}x^T &  &  &   \\&  x^T&  &  \\&  & \ddots  &  \\&  &  & x^T\end{bmatrix}=\begin{bmatrix}z_0x^T\\ \vdots\\ z_mx^T\end{bmatrix}=zx^T$

    Thus $\frac{\partial y}{\partial A}=x^T\in\mathbb{R}^n$
