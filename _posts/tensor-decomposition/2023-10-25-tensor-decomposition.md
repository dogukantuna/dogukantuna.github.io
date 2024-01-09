---
layout: post
title:  "Tensor Decomposition"
date:   2023-10-25 00:00:00 +0000
categories: tensor-decomposition
usemathjax: true
---

**On mathematics**, where abstract constructs often lead to tangible breakthroughs, one operation stands out for its ubiquity and sheer computational appetite: matrix multiplication. This seemingly straightforward task is a cornerstone in a plethora of disciplines — from quantum physics and life sciences to economics and AI. Indeed, optimizing this fundamental mathematical operation promises to unleash transformations that could ripple through not just scientific research but human civilization itself. It's a research frontier that's as exciting as it is daunting, laden with unprecedented opportunities. Mathematicians have been pushing the record of solving the puzzle of faster matrix multiplication since the mid-20th century. In 2022, a research team at Google DeepMind aimed to break a 50-year-old record by discovering faster algorithms for matrix multiplication through training an AI system called **AlphaTensor**.

After the breakthrough results of AlphaTensor were published, mathematicians, recognizing the potential for even further approximations, used conventional computer-aided searches to improve one of the algorithms that the neural network had discovered. The key objective of the trained neural network was to find tensor decompositions within a finite factor space. For the first time, to our knowledge, AlphaTensor's algorithm improves upon Strassen's two-level algorithm, which was discovered 50 years ago. [Here](https://www.quantamagazine.org/ai-reveals-new-possibilities-in-matrix-multiplication-20221123/) is a amazing Quanta Magazine article about that, if you want to know the whole story of decades of research and AlphaTensor.

## Tensor Decomposition: The Basics

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/1_1.png">
</figure>

A tensor, in mathematics, is an algebraic object that describes a multilinear relations between sets of algebraic objects related to a vector space. The order of a tensor is the number of dimensions, also known as ways or modes. In essence, tensor objects are generalization of scalars, vectors, and matrices. Notated as a multidimensional array, or in other words, an N-way array, Nth-order tensor; where 'N' represents the order of the tensor. A first-order tensor essentially defines what we commonly understand as a vector. It has only one axis and can be thought of as a one-dimensional array. A second-order tensor, on the other hand, defines a matrix. It possesses two axes and can be represented as a two-dimensional array. When you move to tensors of order three or higher, these are called higher-order tensors. They have three or more axes and can be conceptualized as arrays with three or more dimensions. 

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/2_2.png">
</figure>

Decompositions of higher-order \\(N\ge 3\\)tensors play a pivotal role in various fields of study and applications. Such decompositions break down a complex tensor into simpler, more manageable parts while retaining its original properties. E.g. CANDECOMP/PARAFAC (CP) decomposition and Tucker decomposition are two particular types that can be considered higher-order extensions of the well-known matrix singular value decomposition (SVD) and as mentioned, the importance of tensor decomposition transcends pure mathematics and finds practical utilities. Many complex mathematical and computational challenges involving tensors can be significantly simplified or even rendered tractable by employing appropriate tensor decompositions. In 1927, Hitchcock put forth the concept of representing a tensor in polyadic form, meaning it could be expressed as the sum of several rank-one tensors. Later, in 1944, Cattell introduced the notions of parallel proportional analysis and multi-dimensional axes for analysis, which include circumstances, objects, and features. Concept finally became popular after its third introduction, in the form of CANDECOMP (canonical decomposition) by Carroll and Chang [38] and PARAFAC
(parallel factors) by Harshman.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/3_3.png">
</figure>

To lay a proper foundation for the variants of tensor decomposition, we can first talk about the core principles and mathematical definitions that encapsulate what tensor decomposition truly is. Mathematically speaking, given a tensor \\(\tau \in R^{n_{1}×n_{2}×...×n_{d}}\\) *(CP form of a tensor)* of order \\(d\\), the goal of tensor decomposition is to approximate \\(\tau\\) as a sum of rank-1 tensors, which is a tensor that can be written as the outer product of \\(d\\) vectors. More formally, our aim is to find vectors \\(a_{i}^k \in R^{n_{k}}\\) for \\(i=1,2,...,R\\) and \\(k=1,2,...,d\\) such that the tensor \\(\tau\\) can be approximated as: \\(\tau ≈ \sum_{i=1}^{R} a_{i}^1 \circ a_{i}^2 \circ ... \circ a_{i}^d\\)

Here, \\(R\\) being the rank of the tensor, and product operator denotes the outer product. Value of \\(R\\) typically determines the quality of the approximation: a lower \\(R\\) often means less complexity and easier computability, but potentially at the cost of approximation accuracy. Contrary to the case of matrices, computing the rank of a tensor is NP-hard problem. This is considered as open problem in computational mathematics.

Specifically, Canonical Polyadic decomposition variant (CPD) can approximate an observed tensor by a sum of rank-1 tensors, which requires \\(O(dnr)\\) parameters and while Tucker decomposition attempts to approximate tensors by a core tensor and several factor matrices, which requires \\(O(dnr + r^d)\\) parameters. Defining feature of the CPD is that it is essentially unique for tensor orders greater than two, allowing for unique parameter identification. mathematically, this can be stated as: for a tensor T of order  N>2, the CP decomposition is essentially unique, facilitating the unambiguous identification of its parameters. There are many algorithms for calculating CP decomposition. The most popular in the literature is the Alternating Least Squares (ALS), which is an iterative optimization process that estimates the factor matrices in an alternating way. 

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/4_4.png">
</figure>

*A canonical polyadic decomposition of a third-order tensor*

However, the challenge with CPD lies in finding the optimal solution. This is because CPD optimization problems are often non-convex, and the search for the best factor matrices can get stuck in local minima. This complexity can make it computationally expensive and time-consuming to determine the ideal CPD representation. On the other hand, the Tucker decomposition provides a more stable and flexible representation of tensors. In Tucker decomposition, a tensor is expressed as a core tensor (also known as the core array) multiplied by a set of orthogonal matrices along each mode.

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/5_5.png">
</figure>

*Canonical Polyadic Decomposition: Robust Manifold Nonnegative Tucker Factorization for Tensor Data Representation - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Canonical-Polyadic-Decomposition_fig1_365209303 [accessed 17 Oct, 2023]*

<figure>
<img src="https://raw.githubusercontent.com/dogukantuna/dogukantuna.github.io/main/_posts/tensor-decomposition/6_6.png">
</figure>

*Tucker Decomposition: Robust Manifold Nonnegative Tucker Factorization for Tensor Data Representation - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Canonical-Polyadic-Decomposition_fig1_365209303 [accessed 17 Oct, 2023]*


There are many tensor decompositions methods including Hierarchical Tucker Decomposition, Block Term Decomposition, Tensor Train Decomposition, INDSCAL,  Matrix Product State, Coupled Matrix/Tensor Factorization, Tensor Ring Decomposition, PARAFAC2, CANDELINC, DEDICOM, and PARATUCK2 as well as nonnegative variants. 

E.g. N-way Toolbox, Tensorly, Tensor Toolbox, and Multilinear Engine are examples of software packages for working with tensor decompositions.

Rich space of matrix multiplication algorithms can be formalized as low-rank decompositions of a specific dim(3) tensors. This algorithmic landscape includes the conventional matrix multiplication algorithm and recursive approaches like Strassen's, in addition to the as-yet-undiscovered algorithm that is asymptotically optimal. Because, in contrast to two-dimensional matrices, for which efficient polynomial-time algorithms computing the rank have existed for over two centuries, finding low-rank decompositions of 3D tensors (and beyond) is NP-hard and is also hard in application. Discovering the most efficient algorithm for matrix multiplication essentially hinges on identifying a decomposition that utilizes the fewest number of rank-1 tensors.

## Variant Routines with Python

Tensorly is package where provide tensor functionalities, decomposition classes, such:

- CP(rank[, n_iter_max, init, svd, ...])

- RandomizedCP(rank, n_samples[, n_iter_max, ...])

- CPPower(rank[, n_repeat, n_iteration, verbose])

- CP_NN_HALS(rank[, n_iter_max, init, svd, ...])

- Tucker([rank, n_iter_max, init, ...])

- TensorTrain(rank[, svd, verbose])

- Parafac2(rank[, n_iter_max, init, svd, ...])

- SymmetricCP(rank[, n_repeat, n_iteration, ...])

- ConstrainedCP(rank[, n_iter_max, ...])

- TensorTrain(rank[, svd, verbose])

- TensorRing(rank[, mode, svd, verbose])

- TensorTrainMatrix(rank[, svd, verbose])


If we make an example PARAFAC decomposition with Python:

```python
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

tensor = np.array(
    [
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [[9, 10], [11, 12], [13, 14], [15, 16]],
        [[17, 18], [19, 20], [21, 22], [23, 24]],
    ]
)

print("Original Tensor:")
print(tensor)

factors = parafac(tensor, rank=2)
tensor_reconstructed = tl.cp_to_tensor(factors)

print("\nReconstructed Tensor:")
print(tensor_reconstructed)

error = tl.norm(tensor - tensor_reconstructed)
print(f"\nReconstruction Error: {error}")
```

PARAFAC computations benefits the Alternating Least Squares (ALS) algorithm (is a very old and reliable method for fitting CP decompositions), calculates the reconstruction errors:

```bash
function PARAFAC(tensor, rank, other parameters...)
    rank = validate_cp_rank(tensor.shape, rank)

    weights, factors = initialize_cp(tensor, rank, other settings...)

    rec_errors = empty_list()
    norm_tensor = norm(tensor)

    for iteration = 1 to n_iter_max
        // Optional: Orthogonalize factors
        if orthogonalize
            factors = orthogonalize_factors(factors)
        
        // Update factors one at a time
        for mode in all_modes_except_fixed
            pseudo_inverse = calculate_pseudo_inverse(factors, weights, other settings...)
            mttkrp = calculate_MTTKRP(tensor, factors, weights, mode)
            
            // Update factor for the current mode
            factors[mode] = solve_linear_equation(pseudo_inverse, mttkrp)

        // Optional: Line search
        if line_search
            perform_line_search()

        // Check convergence
        rec_error = calculate_reconstruction_error(tensor, factors, weights)
        rec_errors.append(rec_error)
        
        if rec_error < tol
            break

    return CPTensor(weights, factors), rec_errors
```

ALS calculation follows the steps:

```bash
Initialize factor matrices
Set iteration count iter = 0
Set prev_error = infinity

while iter < T:
    Fix B, C & update A by solving the LS: 
        A = argmin ||X - A x B x C||^2
    Fix A, C & update B by solving the LS: 
        B = argmin ||X - A x B x C||^2
    Fix A, B & update C by solving the LS: 
        C = argmin ||X - A x B x C||^2
    
    compute_error = ||X - A x B x C||^2
    if abs(prev_error - error) < tol:
           break
    
    prev_error = error
    iter = iter + 1

return A, B, C
```

Achieves the n-mod decomposition of given tensor. ALS iteratively fixes all but one of the factor matrices to solve a least squares problem. Given tensor that is to be decomposed into, R rank-one tensors, the ALS method updates each factor matrix \\(A^(n)\\) while holding other factor matrices constant. Open-source algebraic routine libraries like Tensorly continue to make these complex calculations more accessible. For example we can use the TensorLy library methods to create a synthetic 3D tensor representing temperature variations in s spatial region over different times, altitudes, and lattitudes:

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

from tensorly.random import random_cp
from tensorly.decomposition import CP, parafac
from time import time


def synth_tensor(shape, rank, random_state):
    return random_cp(shape, rank, random_state=random_state, full=True)


def run_parafac(tensor, rank, tol, linesearch):
    start_time = time()
    cp = CP(rank=rank, n_iter_max=2000000, tol=tol, linesearch=linesearch)
    fac = cp.fit_transform(tensor)
    elapsed_time = time() - start_time
    error = tl.norm(tl.cp_to_tensor(fac) - tensor)
    return elapsed_time, error


def plot_res(time_no_ls, error_no_ls, time_ls, error_ls, error_min):
    fig, ax = plt.subplots()
    ax.loglog(time_no_ls, error_no_ls - error_min, ".", label="No line")
    ax.loglog(time_ls, error_ls - error_min, ".r", label="Line")
    ax.legend()
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Error")
    ax.set_title("Time vs Error for Synth Temperature Data")
    plt.show()


def main():
    tensor_shape = (20, 30, 40)
    rank = 5

    tensor = synth_tensor(tensor_shape, rank, random_state=1234)
    tol_values = np.logspace(-1, -9)
    time_no_ls = np.empty_like(tol_values)
    error_no_ls = np.empty_like(tol_values)
    time_ls = np.empty_like(tol_values)
    error_ls = np.empty_like(tol_values)

    _, ref_error = run_parafac(tensor, rank, tol=1.0e-15, linesearch=True)

    for i, tol in enumerate(tol_values):
        time_no_ls[i], error_no_ls[i] = run_parafac(tensor, rank, tol, linesearch=False)
        time_ls[i], error_ls[i] = run_parafac(tensor, rank, tol, linesearch=True)

    plot_res(time_no_ls, error_no_ls, time_ls, error_ls, ref_error)


if __name__ == "__main__":
    main()
```

## References

- Tensor Ring Decomposition - https://arxiv.org/pdf/1606.05535.pdf
- Tensor Decompositions and Applications - https://www.kolda.net/publication/TensorReview.pdf
