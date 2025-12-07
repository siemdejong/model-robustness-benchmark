"""
===========================================
Sensitivity Analysis with Image Augmentations
===========================================

Uses python-adaptive to explore how gamma and brightness transformations
affect the accuracy of a trained MLP classifier on MNIST.

Gamma transformation: I_out = I_in^gamma
Brightness transformation: I_out = I_in + brightness (clipped to [0, 1])

The adaptive sampling intelligently explores the 2D parameter space
to find regions where accuracy changes most rapidly.
"""

import argparse
import os

import adaptive
import numpy as np
from joblib import load
from tqdm import tqdm

from sklearn.datasets import fetch_openml, get_data_home
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array


# Cache directory for classifiers
DEFAULT_CACHE_DIR = os.path.join(get_data_home(), "mnist_classifier_cache")

# Globals for multiprocessing compatibility
GLOBAL_X_TEST = None
GLOBAL_Y_TEST = None
GLOBAL_MODEL = None


def load_test_data(dtype=np.float32, order="C"):
    """Load MNIST test data"""
    print("Loading MNIST dataset...")
    data = fetch_openml("mnist_784", as_frame=True)
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features to [0, 1]
    X = X / 255

    # Use test split (last 10,000 samples)
    n_train = 60000
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_test, y_test


def apply_gamma_transform(X, gamma):
    """Apply gamma transformation: I_out = I_in^gamma"""
    # Clip gamma to avoid numerical issues
    gamma = np.clip(gamma, 0.1, 10.0)
    return np.power(X, gamma)


def apply_brightness_transform(X, brightness):
    """Apply brightness transformation: I_out = I_in + brightness"""
    return np.clip(X + brightness, 0.0, 1.0)


def evaluate_accuracy(X_test, y_test, model, gamma, brightness):
    """
    Evaluate model accuracy after applying gamma and brightness transforms.

    Parameters
    ----------
    X_test : array-like
        Test images (normalized to [0, 1])
    y_test : array-like
        Test labels
    model : sklearn estimator
        Trained classifier
    gamma : float
        Gamma transformation parameter (1.0 = no change)
    brightness : float
        Brightness offset (-1 to 1, 0 = no change)

    Returns
    -------
    accuracy : float
        Classification accuracy on transformed images
    """
    # Apply transformations
    X_transformed = apply_gamma_transform(X_test, gamma)
    X_transformed = apply_brightness_transform(X_transformed, brightness)

    # Predict and compute accuracy
    y_pred = model.predict(X_transformed)
    return accuracy_score(y_test, y_pred)



# Top-level function for multiprocessing
def accuracy_func(params):
    gamma, brightness = params
    return evaluate_accuracy(GLOBAL_X_TEST, GLOBAL_Y_TEST, GLOBAL_MODEL, gamma, brightness)


def run_adaptive_sampling(
    model,
    X_test,
    y_test,
    gamma_bounds=(0.2, 3.0),
    brightness_bounds=(-0.5, 0.5),
    npoints_goal=200,
    n_workers=1,
):
    """
    Run adaptive sampling to explore the parameter space.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier
    X_test : array-like
        Test images
    y_test : array-like
        Test labels
    gamma_bounds : tuple
        (min, max) for gamma parameter
    brightness_bounds : tuple
        (min, max) for brightness parameter
    npoints_goal : int
        Number of points to sample

    Returns
    -------
    learner : adaptive.Learner2D
        The trained learner with sampled data
    """
    import concurrent.futures
    import asyncio


    # Set globals for multiprocessing
    global GLOBAL_X_TEST, GLOBAL_Y_TEST, GLOBAL_MODEL
    GLOBAL_X_TEST = X_test
    GLOBAL_Y_TEST = y_test
    GLOBAL_MODEL = model

    learner = adaptive.Learner2D(
        accuracy_func, bounds=[gamma_bounds, brightness_bounds]
    )

    print(f"Starting adaptive sampling (parallel)...")
    print(f"  Gamma range: {gamma_bounds}")
    print(f"  Brightness range: {brightness_bounds}")
    print(f"  Target points: {npoints_goal}")
    print(f"  Number of workers: {n_workers}")
    print()

    # Use adaptive.Runner for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        runner = adaptive.Runner(
            learner,
            goal=lambda l: l.npoints >= npoints_goal,
            executor=executor,
            shutdown_executor=True,
        )
        # tqdm progress bar integration
        pbar = tqdm(total=npoints_goal, desc="Sampling", unit="pts")
        last_n = 0
        async def progress_watcher():
            while learner.npoints < npoints_goal:
                await asyncio.sleep(0.5)
                pbar.update(learner.npoints - pbar.n)
            pbar.update(npoints_goal - pbar.n)
            pbar.close()
        loop = runner.ioloop
        tasks = [runner.task, progress_watcher()]
        loop.run_until_complete(asyncio.gather(*tasks))

    print(f"Sampling complete! Total points: {learner.npoints}")
    return learner


def analyze_results(learner):
    """Analyze and summarize the adaptive sampling results"""
    data = learner.to_numpy()
    points = data[:, :2]  # (gamma, brightness) pairs
    accuracies = data[:, 2]  # accuracy values

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal samples: {len(accuracies)}")
    print(f"\nAccuracy Statistics:")
    print(f"  Mean accuracy:   {np.mean(accuracies):.4f}")
    print(f"  Std accuracy:    {np.std(accuracies):.4f}")
    print(f"  Min accuracy:    {np.min(accuracies):.4f}")
    print(f"  Max accuracy:    {np.max(accuracies):.4f}")

    # Find best and worst parameter combinations
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)

    print(f"\nBest parameters:")
    print(f"  Gamma: {points[best_idx, 0]:.3f}, Brightness: {points[best_idx, 1]:.3f}")
    print(f"  Accuracy: {accuracies[best_idx]:.4f}")

    print(f"\nWorst parameters:")
    print(f"  Gamma: {points[worst_idx, 0]:.3f}, Brightness: {points[worst_idx, 1]:.3f}")
    print(f"  Accuracy: {accuracies[worst_idx]:.4f}")

    # Analyze sensitivity to each parameter
    # Find points near gamma=1 (no gamma transform)
    gamma_sensitivity_mask = np.abs(points[:, 1]) < 0.1  # brightness near 0
    if np.sum(gamma_sensitivity_mask) > 2:
        gamma_points = points[gamma_sensitivity_mask]
        gamma_accs = accuracies[gamma_sensitivity_mask]
        sort_idx = np.argsort(gamma_points[:, 0])
        gamma_range = gamma_points[sort_idx[-1], 0] - gamma_points[sort_idx[0], 0]
        acc_range = np.max(gamma_accs) - np.min(gamma_accs)
        print(f"\nGamma sensitivity (brightness≈0):")
        print(f"  Accuracy range: {acc_range:.4f} over gamma range {gamma_range:.2f}")

    # Find points near brightness=0 (no brightness transform)
    brightness_sensitivity_mask = np.abs(points[:, 0] - 1.0) < 0.2  # gamma near 1
    if np.sum(brightness_sensitivity_mask) > 2:
        brightness_points = points[brightness_sensitivity_mask]
        brightness_accs = accuracies[brightness_sensitivity_mask]
        sort_idx = np.argsort(brightness_points[:, 1])
        bright_range = (
            brightness_points[sort_idx[-1], 1] - brightness_points[sort_idx[0], 1]
        )
        acc_range = np.max(brightness_accs) - np.min(brightness_accs)
        print(f"\nBrightness sensitivity (gamma≈1):")
        print(
            f"  Accuracy range: {acc_range:.4f} over brightness range {bright_range:.2f}"
        )

    return data


def save_results(learner, output_path):
    """Save the adaptive sampling results to a file"""
    data = learner.to_numpy()
    np.savez(
        output_path,
        gamma=data[:, 0],
        brightness=data[:, 1],
        accuracy=data[:, 2],
    )
    print(f"\nResults saved to: {output_path}")


def plot_results(learner, output_path=None):
    """Plot the adaptive sampling results using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    data = learner.to_numpy()
    gamma = data[:, 0]
    brightness = data[:, 1]
    accuracy = data[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Define colorbar ticks from 0 to 1 in steps of 0.1
    cbar_ticks = np.arange(0, 1.1, 0.1)

    # Scatter plot of sampled points colored by accuracy
    ax1 = axes[0]
    scatter = ax1.scatter(gamma, brightness, c=accuracy, cmap="viridis", s=20, vmin=0, vmax=1)
    ax1.set_xlabel("Gamma")
    ax1.set_ylabel("Brightness")
    ax1.set_title("Sampled Points (colored by accuracy)")
    cbar1 = plt.colorbar(scatter, ax=ax1, label="Accuracy", ticks=cbar_ticks)

    # Interpolated heatmap using tricontourf
    ax2 = axes[1]
    try:
        import matplotlib.tri as tri

        triang = tri.Triangulation(gamma, brightness)
        contour = ax2.tricontourf(triang, accuracy, levels=np.linspace(0, 1, 21), cmap="viridis", vmin=0, vmax=1)
        ax2.set_xlabel("Gamma")
        ax2.set_ylabel("Brightness")
        ax2.set_title("Accuracy Landscape (interpolated)")
        cbar2 = plt.colorbar(contour, ax=ax2, label="Accuracy", ticks=cbar_ticks)
    except Exception as e:
        ax2.text(
            0.5,
            0.5,
            f"Could not create contour:\n{e}",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # Histogram of accuracy distribution
    ax3 = axes[2]
    ax3.hist(accuracy, bins=20, edgecolor="black", alpha=0.7)
    ax3.axvline(np.mean(accuracy), color="red", linestyle="--", label="Mean")
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Count")
    ax3.set_title("Accuracy Distribution")
    ax3.set_xlim(0, 1)
    ax3.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def plot_transform_examples(X_test, gamma_range, brightness_range, output_path=None):
    """
    Plot example images showing how gamma and brightness transforms affect the input.
    
    Creates a grid showing:
    - Original images for a few sample digits
    - Transformed versions across different gamma/brightness combinations
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping transform examples plot")
        return

    # Just use first 5 images for simplicity
    sample_indices = [0, 100, 200, 300, 400]
    n_samples = len(sample_indices)
    
    # Define gamma and brightness values to show
    gamma_values = [gamma_range[0], 0.5, 1.0, 2.0, gamma_range[1]]
    brightness_values = [brightness_range[0], -0.25, 0.0, 0.25, brightness_range[1]]
    n_transforms = len(gamma_values)
    
    # Create two separate figures
    # --- Figure 1: Gamma transformation examples ---
    fig1, axes1 = plt.subplots(n_transforms, n_samples, figsize=(12, 12))
    fig1.suptitle("Gamma Transformation Effects (γ)", fontsize=14, fontweight='bold')
    
    for i, gamma in enumerate(gamma_values):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes1[i, j]
            
            # Get and transform image
            img = X_test[sample_idx].reshape(28, 28)
            img_transformed = apply_gamma_transform(img, gamma)
            
            ax.imshow(img_transformed, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f'γ={gamma:.2f}', fontsize=10, rotation=0, ha='right', va='center')
            if i == 0:
                ax.set_title(f'Sample {j+1}', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        base_name = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
        gamma_path = f"{base_name}_gamma.{ext}"
        plt.savefig(gamma_path, dpi=150, bbox_inches="tight")
        print(f"Gamma transform examples saved to: {gamma_path}")
    
    plt.close(fig1)
    
    # --- Figure 2: Brightness transformation examples ---
    fig2, axes2 = plt.subplots(n_transforms, n_samples, figsize=(12, 12))
    fig2.suptitle("Brightness Transformation Effects (b)", fontsize=14, fontweight='bold')
    
    for i, brightness in enumerate(brightness_values):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes2[i, j]
            
            # Get and transform image
            img = X_test[sample_idx].reshape(28, 28)
            img_transformed = apply_brightness_transform(img, brightness)
            
            ax.imshow(img_transformed, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f'b={brightness:+.2f}', fontsize=10, rotation=0, ha='right', va='center')
            if i == 0:
                ax.set_title(f'Sample {j+1}', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        brightness_path = f"{base_name}_brightness.{ext}"
        plt.savefig(brightness_path, dpi=150, bbox_inches="tight")
        print(f"Brightness transform examples saved to: {brightness_path}")
    else:
        plt.show()
    
    plt.close(fig2)


def plot_pixel_distributions(X_test, gamma_range, brightness_range, output_path=None):
    """
    Plot the distribution of pixel values under different transformations.
    
    Shows how gamma and brightness transforms change the pixel intensity distribution.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping pixel distribution plot")
        return

    # Use a subset of images for efficiency
    X_subset = X_test[:1000].flatten()
    
    # Define transform values
    gamma_values = [gamma_range[0], 0.5, 1.0, 2.0, gamma_range[1]]
    brightness_values = [brightness_range[0], -0.25, 0.0, 0.25, brightness_range[1]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- Gamma transformation distributions ---
    ax1 = axes[0, 0]
    for gamma in gamma_values:
        X_transformed = apply_gamma_transform(X_subset, gamma)
        ax1.hist(X_transformed, bins=50, alpha=0.5, label=f'γ={gamma:.2f}', density=True)
    ax1.set_xlabel('Pixel Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title('Pixel Distribution: Gamma Transform')
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # --- Brightness transformation distributions ---
    ax2 = axes[0, 1]
    for brightness in brightness_values:
        X_transformed = apply_brightness_transform(X_subset, brightness)
        ax2.hist(X_transformed, bins=50, alpha=0.5, label=f'b={brightness:+.2f}', density=True)
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title('Pixel Distribution: Brightness Transform')
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    # --- Combined transform at corners of parameter space ---
    ax3 = axes[1, 0]
    corners = [
        (gamma_range[0], brightness_range[0], 'Low γ, Low b'),
        (gamma_range[0], brightness_range[1], 'Low γ, High b'),
        (gamma_range[1], brightness_range[0], 'High γ, Low b'),
        (gamma_range[1], brightness_range[1], 'High γ, High b'),
        (1.0, 0.0, 'Original'),
    ]
    for gamma, brightness, label in corners:
        X_transformed = apply_gamma_transform(X_subset, gamma)
        X_transformed = apply_brightness_transform(X_transformed, brightness)
        ax3.hist(X_transformed, bins=50, alpha=0.5, label=label, density=True)
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Density')
    ax3.set_title('Pixel Distribution: Combined Transforms (Corners)')
    ax3.legend()
    ax3.set_xlim(0, 1)
    
    # --- Mean pixel intensity heatmap across parameter space ---
    ax4 = axes[1, 1]
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], 20)
    brightness_grid = np.linspace(brightness_range[0], brightness_range[1], 20)
    mean_intensity = np.zeros((len(brightness_grid), len(gamma_grid)))
    
    for i, brightness in enumerate(brightness_grid):
        for j, gamma in enumerate(gamma_grid):
            X_transformed = apply_gamma_transform(X_subset, gamma)
            X_transformed = apply_brightness_transform(X_transformed, brightness)
            mean_intensity[i, j] = np.mean(X_transformed)
    
    im = ax4.imshow(mean_intensity, extent=[gamma_range[0], gamma_range[1], 
                                             brightness_range[0], brightness_range[1]],
                    origin='lower', aspect='auto', cmap='viridis')
    ax4.set_xlabel('Gamma')
    ax4.set_ylabel('Brightness')
    ax4.set_title('Mean Pixel Intensity Across Parameter Space')
    plt.colorbar(im, ax=ax4, label='Mean Intensity')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Pixel distribution plot saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis of MLP classifier to image augmentations"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="MLP-adam",
        help="Name of the cached classifier to load",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory where classifiers are cached",
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=0.2,
        help="Minimum gamma value to explore",
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=3.0,
        help="Maximum gamma value to explore",
    )
    parser.add_argument(
        "--brightness-min",
        type=float,
        default=-0.5,
        help="Minimum brightness offset to explore",
    )
    parser.add_argument(
        "--brightness-max",
        type=float,
        default=0.5,
        help="Maximum brightness offset to explore",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=200,
        help="Number of points to sample adaptively",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for adaptive sampling (python-adaptive)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity_results.npz",
        help="Output file for results (numpy format)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output file for plot (e.g., sensitivity_plot.png)",
    )
    parser.add_argument(
        "--plot-transforms",
        type=str,
        default=None,
        help="Output file for transform visualization (e.g., transforms.png)",
    )

    args = parser.parse_args()

    # Load the cached classifier
    cache_path = os.path.join(args.cache_dir, f"{args.classifier}.joblib")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached classifier not found at {cache_path}. "
            f"Run mnist_robustness.py first to train and cache the classifier."
        )

    print(f"Loading cached classifier: {args.classifier}")
    model = load(cache_path)

    # Load test data
    X_test, y_test = load_test_data()

    # Evaluate baseline accuracy (no transformation)
    baseline_acc = evaluate_accuracy(X_test, y_test, model, gamma=1.0, brightness=0.0)
    print(f"\nBaseline accuracy (gamma=1, brightness=0): {baseline_acc:.4f}")

    import time
    start_time = time.time()
    # Run adaptive sampling
    learner = run_adaptive_sampling(
        model,
        X_test,
        y_test,
        gamma_bounds=(args.gamma_min, args.gamma_max),
        brightness_bounds=(args.brightness_min, args.brightness_max),
        npoints_goal=args.npoints,
        n_workers=args.n_workers,
    )
    elapsed = time.time() - start_time
    print(f"\nElapsed time for adaptive sampling: {elapsed:.2f} seconds\n")

    # Analyze and display results
    analyze_results(learner)

    # Save results
    save_results(learner, args.output)

    # Plot results if requested
    if args.plot:
        plot_results(learner, args.plot)

    # Plot transform visualizations if requested
    if args.plot_transforms:
        # Generate base name for multiple plots
        base_name = args.plot_transforms.rsplit('.', 1)[0]
        ext = args.plot_transforms.rsplit('.', 1)[1] if '.' in args.plot_transforms else 'png'
        
        # Plot example images with transforms (creates _gamma and _brightness files)
        plot_transform_examples(
            X_test,
            gamma_range=(args.gamma_min, args.gamma_max),
            brightness_range=(args.brightness_min, args.brightness_max),
            output_path=args.plot_transforms
        )
        
        # Plot pixel distributions
        plot_pixel_distributions(
            X_test,
            gamma_range=(args.gamma_min, args.gamma_max),
            brightness_range=(args.brightness_min, args.brightness_max),
            output_path=f"{base_name}_distributions.{ext}"
        )
