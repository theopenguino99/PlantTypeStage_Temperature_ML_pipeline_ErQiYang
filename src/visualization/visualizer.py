import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    """Class for creating visualizations."""

    def __init__(self):
        """Initialize the Visualizer."""
        sns.set(style="whitegrid")

    def plot_feature_importance(self, feature_importances, feature_names, top_n=20):
        """
        Plot the feature importances.

        Args:
            feature_importances (array-like): Feature importances
            feature_names (list): List of feature names
            top_n (int): Number of top features to plot
        """
        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Top {} Feature Importances'.format(top_n))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    def plot_predictions(self, y_true, y_pred):
        """
        Plot the true vs predicted values.

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.show()