"""
모델 평가 및 시각화 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """모델 평가"""

    def __init__(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Args:
            predictions: 예측값 배열
            actuals: 실제값 배열
        """
        self.predictions = predictions.flatten()
        self.actuals = actuals.flatten()
        self.metrics = {}

    def calculate_metrics(self) -> Dict[str, float]:
        """평가 지표 계산"""
        # MSE, RMSE
        mse = np.mean((self.actuals - self.predictions) ** 2)
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(self.actuals - self.predictions))

        # MAPE
        mape = np.mean(
            np.abs((self.actuals - self.predictions) / (self.actuals + 1e-8))
        ) * 100

        # R² Score
        ss_res = np.sum((self.actuals - self.predictions) ** 2)
        ss_tot = np.sum((self.actuals - np.mean(self.actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # RMSLE (Root Mean Squared Logarithmic Error)
        rmsle = np.sqrt(
            np.mean((np.log(self.actuals + 1) - np.log(self.predictions + 1)) ** 2)
        )

        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'rmsle': rmsle,
        }

        logger.info(f"\n평가 지표:")
        logger.info(f"  RMSE:  {rmse:.6f}")
        logger.info(f"  MAE:   {mae:.6f}")
        logger.info(f"  MAPE:  {mape:.4f}%")
        logger.info(f"  R²:    {r2:.4f}")
        logger.info(f"  RMSLE: {rmsle:.6f}")

        return self.metrics

    def get_residuals(self) -> np.ndarray:
        """잔차 반환"""
        return self.actuals - self.predictions

    def get_metrics(self) -> Dict[str, float]:
        """지표 반환"""
        if not self.metrics:
            self.calculate_metrics()
        return self.metrics


class PredictionAnalyzer:
    """예측 분석"""

    def __init__(self, dates: pd.Series, predictions: np.ndarray,
                 actuals: np.ndarray, normalizer=None):
        """
        Args:
            dates: 날짜 시리즈
            predictions: 예측값
            actuals: 실제값
            normalizer: 역정규화기 (선택)
        """
        self.dates = dates.values
        self.predictions = predictions.flatten()
        self.actuals = actuals.flatten()
        self.normalizer = normalizer

    def inverse_normalize(self, column_name: str = 'total_sales') -> Tuple[np.ndarray, np.ndarray]:
        """역정규화"""
        if self.normalizer is None:
            return self.predictions, self.actuals

        pred_inv = self.normalizer.inverse_transform(
            pd.DataFrame({column_name: self.predictions})
        ).flatten()

        actual_inv = self.normalizer.inverse_transform(
            pd.DataFrame({column_name: self.actuals})
        ).flatten()

        return pred_inv, actual_inv

    def create_forecast_dataframe(self, column_name: str = 'total_sales') -> pd.DataFrame:
        """예측 데이터프레임 생성"""
        pred_inv, actual_inv = self.inverse_normalize(column_name)

        df = pd.DataFrame({
            'date': self.dates,
            'actual': actual_inv,
            'prediction': pred_inv,
            'error': actual_inv - pred_inv,
            'error_pct': np.abs(actual_inv - pred_inv) / (actual_inv + 1e-8) * 100,
        })

        return df

    def get_forecast_summary(self) -> Dict:
        """예측 요약"""
        pred_inv, actual_inv = self.inverse_normalize()

        summary = {
            'actual_mean': np.mean(actual_inv),
            'prediction_mean': np.mean(pred_inv),
            'actual_std': np.std(actual_inv),
            'prediction_std': np.std(pred_inv),
            'mean_error': np.mean(actual_inv - pred_inv),
            'mean_abs_error': np.mean(np.abs(actual_inv - pred_inv)),
        }

        return summary


class VisualizationHelper:
    """시각화 도우미"""

    @staticmethod
    def plot_training_history(history: Dict, figsize: Tuple = (12, 4)) -> plt.Figure:
        """학습 이력 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Loss
        if 'loss' in history:
            ax1.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.set_title('Model Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # MAE
        if 'mae' in history:
            ax2.plot(history['mae'], label='Train MAE')
            if 'val_mae' in history:
                ax2.plot(history['val_mae'], label='Val MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title('Model MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_predictions(dates: pd.Series, actuals: np.ndarray,
                        predictions: np.ndarray, figsize: Tuple = (14, 5)) -> plt.Figure:
        """예측값 vs 실제값 시각화"""
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(dates, actuals, label='Actual', marker='o', linewidth=2, alpha=0.7)
        ax.plot(dates, predictions, label='Prediction', marker='s', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.set_title('Actual vs Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_residuals(residuals: np.ndarray, figsize: Tuple = (12, 5)) -> plt.Figure:
        """잔차 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 잔차 분포
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Residuals')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residuals Distribution')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.grid(True, alpha=0.3)

        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_error_over_time(dates: pd.Series, errors: np.ndarray,
                            figsize: Tuple = (14, 4)) -> plt.Figure:
        """시간별 오류 시각화"""
        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(dates, errors, alpha=0.7, color='steelblue')
        ax.axhline(y=0, color='r', linestyle='-', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Error')
        ax.set_title('Prediction Error Over Time')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_scatter(actuals: np.ndarray, predictions: np.ndarray,
                    figsize: Tuple = (8, 6)) -> plt.Figure:
        """예측 정확도 산점도"""
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(actuals, predictions, alpha=0.5, s=50)

        # 완벽한 예측 라인
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('Actual')
        ax.set_ylabel('Prediction')
        ax.set_title('Prediction Accuracy')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Evaluator 모듈 정의 완료")
