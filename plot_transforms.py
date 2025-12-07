"""
===========================================
Plot Transform Effects on MNIST Images
===========================================

Visualize how gamma and brightness transformations affect MNIST images.
Use this script to explore appropriate parameter ranges before running
the full sensitivity analysis.

Gamma transformation: I_out = I_in^gamma
  - gamma < 1: brightens mid-tones
  - gamma = 1: no change
  - gamma > 1: darkens mid-tones

Brightness transformation: I_out = I_in + brightness (clipped to [0, 1])
  - brightness < 0: darkens image
  - brightness = 0: no change
  - brightness > 0: brightens image

Example usage:
    # Explore default ranges
    uv run plot_transforms.py

    # Custom gamma range
    uv run plot_transforms.py --gamma-min 0.5 --gamma-max 2.0

    # Save plots to files
    uv run plot_transforms.py --output transforms.png

    # Specify exact gamma and brightness values to visualize
    uv run plot_transforms.py --gamma-values 0.3 0.5 1.0 2.0 3.0 \
                               --brightness-values -0.4 -0.2 0.0 0.2 0.4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import check_array


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
    gamma = np.clip(gamma, 0.1, 10.0)
    return np.power(X, gamma)


def apply_brightness_transform(X, brightness):
    """Apply brightness transformation: I_out = I_in + brightness"""
    return np.clip(X + brightness, 0.0, 1.0)


def plot_transform_examples(X_test, gamma_values, brightness_values, 
                            sample_indices=None, output_path=None):
    """
    Plot example images showing how gamma and brightness transforms affect the input.
    """
    if sample_indices is None:
        sample_indices = [0, 100, 200, 300, 400]
    
    n_samples = len(sample_indices)
    n_gamma = len(gamma_values)
    n_brightness = len(brightness_values)
    
    # --- Figure 1: Gamma transformation examples ---
    fig1, axes1 = plt.subplots(n_gamma, n_samples, figsize=(2.5 * n_samples, 2.5 * n_gamma))
    fig1.suptitle("Gamma Transformation Effects (γ)", fontsize=14, fontweight='bold')
    
    # Handle single row case
    if n_gamma == 1:
        axes1 = axes1.reshape(1, -1)
    
    for i, gamma in enumerate(gamma_values):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes1[i, j]
            
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
    else:
        plt.show()
    
    plt.close(fig1)
    
    # --- Figure 2: Brightness transformation examples ---
    fig2, axes2 = plt.subplots(n_brightness, n_samples, figsize=(2.5 * n_samples, 2.5 * n_brightness))
    fig2.suptitle("Brightness Transformation Effects (b)", fontsize=14, fontweight='bold')
    
    if n_brightness == 1:
        axes2 = axes2.reshape(1, -1)
    
    for i, brightness in enumerate(brightness_values):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes2[i, j]
            
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


def plot_combined_transforms(X_test, gamma_values, brightness_values, 
                              sample_idx=0, output_path=None):
    """
    Plot a grid showing combined gamma and brightness transforms.
    
    This creates a single figure with gamma on one axis and brightness on the other,
    making it easy to see the interaction between the two transforms.
    """
    n_gamma = len(gamma_values)
    n_brightness = len(brightness_values)
    
    fig, axes = plt.subplots(n_brightness, n_gamma, 
                             figsize=(2 * n_gamma, 2 * n_brightness))
    fig.suptitle(f"Combined Transforms (Sample {sample_idx})", fontsize=14, fontweight='bold')
    
    # Handle edge cases
    if n_brightness == 1 and n_gamma == 1:
        axes = np.array([[axes]])
    elif n_brightness == 1:
        axes = axes.reshape(1, -1)
    elif n_gamma == 1:
        axes = axes.reshape(-1, 1)
    
    img = X_test[sample_idx].reshape(28, 28)
    
    for i, brightness in enumerate(brightness_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i, j]
            
            # Apply both transforms
            img_transformed = apply_gamma_transform(img, gamma)
            img_transformed = apply_brightness_transform(img_transformed, brightness)
            
            ax.imshow(img_transformed, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f'b={brightness:+.2f}', fontsize=9, rotation=0, ha='right', va='center')
            if i == 0:
                ax.set_title(f'γ={gamma:.2f}', fontsize=9)
    
    # Add axis labels
    fig.text(0.5, 0.02, 'Gamma (γ)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Brightness (b)', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    if output_path:
        base_name = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
        combined_path = f"{base_name}_combined.{ext}"
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        print(f"Combined transform grid saved to: {combined_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_pixel_distributions(X_test, gamma_values, brightness_values, output_path=None):
    """
    Plot the distribution of pixel values under different transformations.
    """
    # Use a subset of images for efficiency
    X_subset = X_test[:1000].flatten()
    
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
    gamma_min, gamma_max = min(gamma_values), max(gamma_values)
    brightness_min, brightness_max = min(brightness_values), max(brightness_values)
    corners = [
        (gamma_min, brightness_min, f'γ={gamma_min:.1f}, b={brightness_min:+.1f}'),
        (gamma_min, brightness_max, f'γ={gamma_min:.1f}, b={brightness_max:+.1f}'),
        (gamma_max, brightness_min, f'γ={gamma_max:.1f}, b={brightness_min:+.1f}'),
        (gamma_max, brightness_max, f'γ={gamma_max:.1f}, b={brightness_max:+.1f}'),
        (1.0, 0.0, 'Original'),
    ]
    for gamma, brightness, label in corners:
        X_transformed = apply_gamma_transform(X_subset, gamma)
        X_transformed = apply_brightness_transform(X_transformed, brightness)
        ax3.hist(X_transformed, bins=50, alpha=0.5, label=label, density=True)
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Density')
    ax3.set_title('Pixel Distribution: Combined Transforms (Corners)')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 1)
    
    # --- Mean pixel intensity heatmap across parameter space ---
    ax4 = axes[1, 1]
    gamma_grid = np.linspace(gamma_min, gamma_max, 20)
    brightness_grid = np.linspace(brightness_min, brightness_max, 20)
    mean_intensity = np.zeros((len(brightness_grid), len(gamma_grid)))
    
    for i, brightness in enumerate(brightness_grid):
        for j, gamma in enumerate(gamma_grid):
            X_transformed = apply_gamma_transform(X_subset, gamma)
            X_transformed = apply_brightness_transform(X_transformed, brightness)
            mean_intensity[i, j] = np.mean(X_transformed)
    
    im = ax4.imshow(mean_intensity, extent=[gamma_min, gamma_max, 
                                             brightness_min, brightness_max],
                    origin='lower', aspect='auto', cmap='viridis')
    ax4.set_xlabel('Gamma')
    ax4.set_ylabel('Brightness')
    ax4.set_title('Mean Pixel Intensity Across Parameter Space')
    plt.colorbar(im, ax=ax4, label='Mean Intensity')
    
    plt.tight_layout()
    
    if output_path:
        base_name = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
        dist_path = f"{base_name}_distributions.{ext}"
        plt.savefig(dist_path, dpi=150, bbox_inches="tight")
        print(f"Pixel distribution plot saved to: {dist_path}")
    else:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gamma and brightness transform effects on MNIST images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View with default ranges
  uv run plot_transforms.py

  # Custom gamma range (will generate 5 evenly spaced values)
  uv run plot_transforms.py --gamma-min 0.5 --gamma-max 2.0

  # Specify exact values to visualize
  uv run plot_transforms.py --gamma-values 0.3 0.5 1.0 2.0 3.0

  # Save to files instead of displaying
  uv run plot_transforms.py --output transforms.png

  # Show combined transform grid for a specific sample
  uv run plot_transforms.py --combined --sample 42
        """
    )
    
    # Gamma arguments
    gamma_group = parser.add_argument_group('Gamma transform options')
    gamma_group.add_argument(
        "--gamma-min",
        type=float,
        default=0.2,
        help="Minimum gamma value (default: 0.2)",
    )
    gamma_group.add_argument(
        "--gamma-max",
        type=float,
        default=3.0,
        help="Maximum gamma value (default: 3.0)",
    )
    gamma_group.add_argument(
        "--gamma-values",
        type=float,
        nargs="+",
        default=None,
        help="Specific gamma values to visualize (overrides min/max)",
    )
    
    # Brightness arguments
    brightness_group = parser.add_argument_group('Brightness transform options')
    brightness_group.add_argument(
        "--brightness-min",
        type=float,
        default=-0.5,
        help="Minimum brightness offset (default: -0.5)",
    )
    brightness_group.add_argument(
        "--brightness-max",
        type=float,
        default=0.5,
        help="Maximum brightness offset (default: 0.5)",
    )
    brightness_group.add_argument(
        "--brightness-values",
        type=float,
        nargs="+",
        default=None,
        help="Specific brightness values to visualize (overrides min/max)",
    )
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (e.g., transforms.png). If not specified, displays plots.",
    )
    output_group.add_argument(
        "--combined",
        action="store_true",
        help="Also generate a combined transform grid showing both transforms together",
    )
    output_group.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to use for combined transform grid (default: 0)",
    )
    output_group.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Sample indices to use for individual transform plots (default: 0 100 200 300 400)",
    )
    output_group.add_argument(
        "--no-distributions",
        action="store_true",
        help="Skip pixel distribution plots",
    )

    args = parser.parse_args()

    # Determine gamma values to use
    if args.gamma_values is not None:
        gamma_values = sorted(args.gamma_values)
    else:
        gamma_values = [args.gamma_min, 0.5, 1.0, 2.0, args.gamma_max]
        # Remove duplicates and sort
        gamma_values = sorted(set(gamma_values))
    
    # Determine brightness values to use
    if args.brightness_values is not None:
        brightness_values = sorted(args.brightness_values)
    else:
        brightness_values = [args.brightness_min, -0.25, 0.0, 0.25, args.brightness_max]
        brightness_values = sorted(set(brightness_values))
    
    # Sample indices
    sample_indices = args.samples if args.samples else [0, 100, 200, 300, 400]

    print(f"Gamma values: {gamma_values}")
    print(f"Brightness values: {brightness_values}")
    print()

    # Load data
    X_test, y_test = load_test_data()

    # Plot individual transform effects
    print("Generating transform example plots...")
    plot_transform_examples(
        X_test, 
        gamma_values, 
        brightness_values,
        sample_indices=sample_indices,
        output_path=args.output
    )

    # Plot combined transform grid if requested
    if args.combined:
        print("Generating combined transform grid...")
        plot_combined_transforms(
            X_test,
            gamma_values,
            brightness_values,
            sample_idx=args.sample,
            output_path=args.output
        )

    # Plot pixel distributions
    if not args.no_distributions:
        print("Generating pixel distribution plots...")
        plot_pixel_distributions(
            X_test,
            gamma_values,
            brightness_values,
            output_path=args.output
        )

    print("\nDone!")
    if args.output:
        print(f"\nTo run sensitivity analysis with these ranges:")
        print(f"  uv run sensitivity_analysis.py \\")
        print(f"      --gamma-min {min(gamma_values)} --gamma-max {max(gamma_values)} \\")
        print(f"      --brightness-min {min(brightness_values)} --brightness-max {max(brightness_values)}")


if __name__ == "__main__":
    main()
