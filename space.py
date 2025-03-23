import numpy as np
import matplotlib.pyplot as plt

def compute_subspace_angle(face_block, stim_vec, n_components=10):
    """
    face_block: shape (n_neurons, n_time)  e.g. (n_neurons, 1000)
    stim_vec:   shape (n_neurons,)        the "direction" for that stimulus
    n_components: how many principal components to keep in the face subspace
    
    Returns: angle_in_degrees between stim_vec and the subspace spanned
             by the top n_components principal components of face_block.
    """

    # 1) Center the face block across columns
    #    shape still (n_neurons, n_time)
    face_mean = np.mean(face_block, axis=1, keepdims=True)  # (n_neurons, 1)
    face_centered = face_block - face_mean

    # 2) PCA via SVD on (n_neurons x n_time)
    #    U has shape (n_neurons, n_neurons) if full_matrices=True, but we do:
    U, S, Vt = np.linalg.svd(face_centered, full_matrices=False)
    #    => U: (n_neurons, rank), S: (rank,), Vt: (rank, n_time)
    # rank <= min(n_neurons, n_time)

    # 3) Select top k principal components
    k = min(n_components, U.shape[1])  # in case n_components > rank
    subspace = U[:, :k]  # shape (n_neurons, k)

    # 4) Project stim_vec onto that subspace
    #    Optionally, we might center stim_vec, but typically we just use it "as is"
    #    stim_vec: shape (n_neurons,)
    proj = subspace @ (subspace.T @ stim_vec)  # shape (n_neurons,)

    # 5) Compute angle between stim_vec and its projection
    #    angle = arccos( dot(v, p) / (||v|| * ||p||) )
    dot_vp = np.dot(stim_vec, proj)
    norm_v = np.linalg.norm(stim_vec)
    norm_p = np.linalg.norm(proj)

    if norm_v < 1e-12 or norm_p < 1e-12:
        # If either vector is zero, define angle as 90 deg or 0 deg as you prefer
        # We'll define it as 90 degrees if the projection is trivial
        return 90.0

    cos_theta = dot_vp / (norm_v * norm_p)
    # Numerical safety
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = angle_radians * 180.0 / np.pi

    return angle_degrees


def main():
    # 1) Load the two .npy files
    face_code = np.load("pred_no_stim_face_motion_L4_neuron_20201022_Y36_Z360.npy")
    stim_code = np.load("pred_stim_no_face_motion_L4_neuron_20201022_Y36_Z360.npy")

    # face_code: shape (n_neurons, 4000)
    # stim_code: shape (n_neurons, 4)

    n_neurons, total_cols = face_code.shape  # total_cols=4000
    # We assume 4 stimulus conditions => each block is 1000 columns
    block_size = 1000  # face_code columns per stimulus

    # 2) We'll compute an angle for each stimulus i in [0..3]
    angles = []

    for i in range(4):
        # a) Face block for stimulus i => columns [i*1000 : (i+1)*1000]
        start_col = i * block_size
        end_col   = (i + 1) * block_size
        face_block_i = face_code[:, start_col:end_col]  # (n_neurons, 1000)

        # b) The corresponding stim direction => stim_code[:, i]
        #    shape (n_neurons,)
        stim_vec_i = stim_code[:, i]

        # c) Compute angle to subspace
        angle_i = compute_subspace_angle(face_block_i, stim_vec_i, n_components=2)
        angles.append(angle_i)

        print(f"Stimulus {i+1}, angle to face subspace (top 2 PCs): {angle_i:.2f} degrees")

    # 3) Visualization
    # We'll just make a bar plot of these 4 angles
    plt.figure()
    plt.bar(range(1, 5), angles)  # x positions = [1,2,3,4]
    plt.xlabel("Stimulus Condition")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle between Stimulus Direction and Face-Code Subspace")
    plt.show()


if __name__ == "__main__":
    main()
    

import numpy as np
import matplotlib.pyplot as plt

def principal_angles_subspaces(U, V):
    """
    Given two orthonormal bases U (n x k1) and V (n x k2),
    compute the principal angles (in degrees) between them.
    
    Returns a sorted list of angles [theta_1, theta_2, ..., theta_r]
    with r = min(k1, k2), from smallest to largest angle.
    """
    # M = U^T V, shape (k1, k2)
    M = U.T @ V
    # SVD => singular values = cosines of principal angles
    # We only need the singular values
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    
    # Clip s into [0,1], then angles = arccos(s)
    s_clipped = np.clip(s, -1.0, 1.0)
    angles_rad = np.arccos(s_clipped)
    angles_deg = angles_rad * 180.0 / np.pi
    
    # Return from smallest to largest angle
    return np.sort(angles_deg)


def orthonormalize(A):
    """
    Orthonormalize columns of A (n x k) using, e.g., QR decomposition or SVD.
    Returns an orthonormal basis with shape (n, rank).
    """
    # We can use QR: A = Q R
    # Q is orthonormal, shape (n, k)
    Q, R = np.linalg.qr(A, mode='reduced')
    return Q


def get_subspace_basis(data, n_components=None):
    """
    data: shape (n_neurons, N) - columns are samples/vectors
    1) Center data across columns
    2) SVD => U, S, V
    3) Keep top 'n_components' columns of U as basis
    4) Return (n_neurons, rank) orthonormal basis
    """
    # Center across columns
    mean_col = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_col  # shape (n_neurons, N)

    # SVD
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    rank = U.shape[1]  # up to min(n_neurons, N)

    if n_components is None or n_components > rank:
        n_c = rank
    else:
        n_c = n_components

    # U[:, :n_c] => orthonormal basis for top PCs
    return U[:, :n_c]


def main():
    # 1) Load the face- and stimulus-coded data
    face_code = np.load("pred_no_stim_face_motion_L4_neuron_20201022_Y36_Z360.npy")
    stim_code = np.load("pred_stim_no_face_motion_L4_neuron_20201022_Y36_Z360.npy")
    # face_code: shape (n_neurons, 4000)
    # stim_code: shape (n_neurons, 4)

    n_neurons, face_cols = face_code.shape
    # 2) We'll define subspace A from the entire face_code (all 4000 columns)
    #    and subspace B from the entire stim_code (all 4 columns).
    
    # Let's pick how many PCs we want from each. For face_code, we might want ~50 PCs
    # For stim_code, there's only 4 columns total, so the maximum rank is 4 anyway.
    face_subspace_dim = 50
    stim_subspace_dim = 4  # can't exceed 4, but you could do less if you want

    # -- get an orthonormal basis for face_code
    U_face = get_subspace_basis(face_code, n_components=face_subspace_dim)
    # shape (n_neurons, face_subspace_dim)

    # -- get an orthonormal basis for stim_code
    U_stim = get_subspace_basis(stim_code, n_components=stim_subspace_dim)
    # shape (n_neurons, stim_subspace_dim)

    # 3) Compute principal angles
    angles = principal_angles_subspaces(U_face, U_stim)
    # 'angles' is a sorted array, from smallest angle to largest
    # length = min(face_subspace_dim, stim_subspace_dim)

    print(f"Principal angles (degrees) between face subspace (dim={U_face.shape[1]}) "
          f"and stim subspace (dim={U_stim.shape[1]}):")
    for i, ang in enumerate(angles, start=1):
        print(f"  Angle {i}: {ang:.2f}°")

    # 4) Plot these principal angles
    plt.figure()
    plt.bar(range(1, len(angles)+1), angles)
    plt.xlabel("Principal Angle Index")
    plt.ylabel("Angle (degrees)")
    plt.title(f"Subspace Angles: Face (dim={U_face.shape[1]}) vs. Stim (dim={U_stim.shape[1]})")
    plt.show()


if __name__ == "__main__":
    main()