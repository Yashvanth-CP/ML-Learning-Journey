"""
Utility functions for regression model evaluation
"""
import  numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class RegressionMetrics:
    # calculate and display regression performance metrics

    @staticmethod

    def calculate(y_true, y_pred, model_name = "Model"):
        """
        Calculation R^2, RMSE, MAE for prediction
        y_trur : Actual values
        y_pred : Predicted value
        model name : name of the model

        """

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f"📊 {model_name} Performance")
        print(f"{'='*50}")
        print(f"R² Score:  {r2:.4f}  (higher is better, max=1.0)")
        print(f"RMSE: {rmse:.4f} (lower is better)")
        print(f"MAE: {mae:.4f} (lower is better)")
        print(f"MSE: {mse:.4f}")
        print(f"{'='*50}\n")

        return {"R2" : r2, "RMSE": rmse, "MAE": mae, "MSE":mse}

    @staticmethod
    def print_comparition(results_dist):
        # Print comparition of multiple models

        print("\n" + "="*60)
        print("Model comparision")
        print("="*60)

        # Sort by R2 Score (descending)

        sorted_results = sorted(results_dist.item(), key =lambda x:x[1]["R2"],reverse=True)

        print(f"{'Model :<25'}{'R^2':<12} {'RMSE':<12}{'MAE':<12}")

        print("-"*60)

        for model_name, metrics in sorted_results:
            print(f"{model_name:<25}{metrics['R2']:<12.4f}{metrics['RMSE']:<12.4f}{metrics['MAE']:<12.4f}")
            print("-"*60)