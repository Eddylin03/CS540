from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Load the dataset
    dataset = np.load(filename)
    # Center the dataset
    mean = np.mean(dataset, axis=0)
    centered_dataset = dataset - mean
    return centered_dataset
    raise NotImplementedError


def get_covariance(dataset):
    #number of samples
    n = dataset.shape[0]
    # covariance matrix
    s = np.dot(np.transpose(dataset), dataset)/(n-1)
    return s
    raise NotImplementedError

def get_eig(S, m):
    #  # Perform eigendecomposition
    # eigenvalues, eigenvectors = eigh(S)
    # sorted_indices = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[sorted_indices]
    # eigenvectors = eigenvectors[:, sorted_indices]
    # top_eigenvalues = np.diag(eigenvalues[:m])  # Convert to diagonal matrix
    # top_eigenvectors = eigenvectors[:, :m]
    
    # return top_eigenvalues, top_eigenvectors

        # Number of features
    k = S.shape[0]
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[k - m, k - 1])
    # Reverse to get descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    # Create a diagonal matrix of the largest k eigenvalues
    Lambda = np.diag(eigenvalues)
    return Lambda, eigenvectors
    raise NotImplementedError

def get_eig_prop(S, prop):

    # Compute all eigenvalues&vec
    eigenvalues, eigenvectors = eigh(S)
   
    eigenvalues = eigenvalues[::-1] # descending order
    eigenvectors = eigenvectors[:, ::-1] # descending order
    total_variance = np.sum(eigenvalues)
    variance_proportions = eigenvalues / total_variance
    
    indices = np.where(variance_proportions > prop)[0]# Select indices where proportion > prop
    selected_eigenvalues = eigenvalues[indices]
    selected_eigenvectors = eigenvectors[:, indices]
    
    Lambda = np.diag(selected_eigenvalues)# create diagonal matrix
    return Lambda, selected_eigenvectors

    raise NotImplementedError

def project_image(image, U):
    
    projection = np.dot(np.transpose(U), image)#compute projecrion weights
    
    # Reconstruct the image
    reconstructed_image = np.dot(U, projection)
    
    return reconstructed_image

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, ax1, ax2 = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    # Reshape images to 64 x 64
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    # Display original image
    im1 = ax1.imshow(orig_image, aspect='equal')
    ax1.set_title('Original')
    fig.colorbar(im1, ax=ax1)
    # Display projected image
    im2 = ax2.imshow(proj_image, aspect='equal')
    ax2.set_title('Projection')
    fig.colorbar(im2, ax=ax2)
    return fig, ax1, ax2
    raise NotImplementedError

def perturb_image(image, U, sigma):
    alpha = np.dot(U.T, image)
    # Generate perturbation from a Gaussian distribution
    perturbation = np.random.normal(0, sigma, size=alpha.shape)
    # Perturb the weights
    alpha_perturbed = alpha + perturbation
    # Reconstruct the image with perturbed weights
    perturbed_image = np.dot(U, alpha_perturbed)
    return perturbed_image
    raise NotImplementedError

def combine_image(image1, image2, U, lam):
    # Compute the projection weights for both images
    alpha1 = np.dot(U.T, image1)
    alpha2 = np.dot(U.T, image2)
    # Compute the convex combination of the weights
    alpha_combined = lam * alpha1 + (1 - lam) * alpha2
    # Reconstruct the combined image
    combined_image = np.dot(U, alpha_combined)
    return combined_image
    raise NotImplementedError