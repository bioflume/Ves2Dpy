import numpy as np

def ray_casting(y, S):
    p0 = 1.1 * np.max(S) * np.ones(2)  # a point outside S
    ray = np.column_stack((p0, y))
    # V1 and V2 are unchanging, defined from ray
    V1 = ray[:, 0]
    V2 = ray[:, 1]
    # V3 is 2x1xN so each pair of neighbours from S is its own "page"
    V3 = np.reshape(S, (2, 1, -1))
    # V4 is a wrap-around of V3, move all pairs by one page
    V4 = np.roll(V3, -1, axis=2)
    # Compute B in one hit, no loop required now we have paged V1 and V3
    B = V3 - V1[:, np.newaxis, np.newaxis]

    # Right-hand side of A is defined as diff of V3 and V4 as before
    AR = V3 - V4
    # Left-hand side of A was always fixed by ray anyway, quick repmat
    AL = np.repeat((V2 - V1)[:, np.newaxis, np.newaxis], AR.shape[2], -1)
    # Define A, 2x2xN square matrices "paged" for each check
    # where N = 0.5*size(S,1)
    A = np.concatenate((AL, AR), axis=1)

    # check for ill-conditioned matrices from the abs determinant
    # faster to compute the det manually as it's only 2x2 and we can
    # vectorize it this way
    det_A = A[0, 0, :] * A[1, 1, :] - A[0, 1, :] * A[1, 0, :]
    idx = np.abs(det_A) >= 1e-7
    # For all well conditioned indices, compute A\b
    # original matlab function was pagemldivide
    alpha = np.linalg.solve(np.transpose(A[:, :, idx],(2,0,1)), 
                                np.transpose(B[:, :, idx],(2,0,1))).squeeze()
    # Count the number of elements where both rows of alpha are 0<=a<=1
    hit = np.sum((0 <= alpha[0, :]) & (alpha[0, :] <= 1) & (0 <= alpha[1, :]) & (alpha[1, :] <= 1))

    # Output flag
    if hit % 2 == 0:
        return 0
    else:
        return 1
