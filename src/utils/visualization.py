"""Visualization utilities for calibration application."""
import io
import os
import base64
import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use Agg backend - doesn't require GUI
import matplotlib.pyplot as plt
from datetime import datetime


def plot_calibration_curve(predictions, actual_values=None):
    """Generate calibration curve plot."""
    plt.figure(figsize=(10, 6))
    
    sorted_predictions = np.sort(predictions)
    
    quantiles = np.linspace(0, 1, len(sorted_predictions))
    
    # Plot the calibration curve (predictions vs quantiles)
    plt.plot(sorted_predictions, quantiles, 'b-', linewidth=2, label='Calibration Curve')
    
    # If actual values are provided, plot them too
    if actual_values is not None:
        # Filter out None values
        valid_indices = [i for i, val in enumerate(actual_values) if val is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_actuals = [actual_values[i] for i in valid_indices]
        
        if valid_actuals:
            plt.scatter(valid_predictions, valid_actuals, color='red', label='Actual vs Predicted')
    
    # Highlight the most recent prediction
    if predictions:
        latest_pred = predictions[-1]
        # Find where the latest prediction sits on the curve
        idx = np.searchsorted(sorted_predictions, latest_pred)
        if idx < len(quantiles):
            plt.scatter([latest_pred], [quantiles[idx]], color='green', s=100, 
                        label='Current Prediction', zorder=5)
    
    plt.xlabel('Predicted Value')
    plt.ylabel('Quantile / Actual Value')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to a base64 encoded string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url


def save_result_image(fig, filename=None, subdir=None):
    """Save a matplotlib figure to the results directory.
    
    Args:
        fig: A matplotlib figure object to save
        filename: Optional filename, if None a timestamp will be used
        subdir: Optional subdirectory within uploads/results/images
        
    Returns:
        The relative path to the saved image
    """
    # Create a timestamp-based filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}.png"
    
    # Make sure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    # Define base directory
    base_dir = 'uploads/results/images'
    
    # Add subdirectory if specified
    if subdir:
        save_dir = os.path.join(base_dir, subdir)
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = base_dir
    
    # Create full path
    full_path = os.path.join(save_dir, filename)
    
    # Save the figure
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return full_path


def create_heatmap(data, title, xticklabels=None, yticklabels=None, cmap='coolwarm', 
                  save=True, filename=None):
    """Create a heatmap visualization and optionally save it.
    
    Args:
        data: 2D array or DataFrame to visualize
        title: Title for the heatmap
        xticklabels: Labels for x-axis
        yticklabels: Labels for y-axis
        cmap: Colormap to use
        save: Whether to save the image
        filename: Optional filename for saving
        
    Returns:
        The path to the saved image if save=True, otherwise the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set title
    ax.set_title(title)
    
    # Set ticks and labels
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    
    # Add grid lines
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Format nicely
    plt.tight_layout()
    
    if save:
        return save_result_image(fig, filename, subdir='heatmaps')
    else:
        return fig 