"""
모델 학습 관리 모듈
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """모델 학습 관리"""

    def __init__(self, model, model_name: str = "global_model"):
        """
        Args:
            model: LSTM 모델 인스턴스
            model_name: 모델명
        """
        self.model = model
        self.model_name = model_name
        self.train_results = {}

    def prepare_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 준비"""
        from src.models.lstm_model import SequenceDataGenerator

        generator = SequenceDataGenerator(self.model.lookback)
        X, y = generator.create_sequences(data, target)

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              **kwargs) -> Dict:
        """모델 학습"""
        logger.info(f"\n{'='*60}")
        logger.info(f"모델 학습: {self.model_name}")
        logger.info(f"{'='*60}")

        history = self.model.train(X_train, y_train, X_val, y_val, **kwargs)

        # 결과 저장
        self.train_results = {
            'model_name': self.model_name,
            'history': self.model.get_history(),
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
        }

        return self.train_results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """모델 평가"""
        logger.info(f"\n모델 평가: {self.model_name}")

        # 예측
        predictions = self.model.predict(X_test)

        # 메트릭 계산
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        mape = np.mean(np.abs((y_test - predictions.flatten()) / (y_test + 1e-8))) * 100

        # R² 스코어
        ss_res = np.sum((y_test - predictions.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'mse': mse,
        }

        logger.info(f"✓ RMSE: {rmse:.6f}")
        logger.info(f"✓ MAE: {mae:.6f}")
        logger.info(f"✓ MAPE: {mape:.4f}%")
        logger.info(f"✓ R²: {r2:.4f}")

        return metrics, predictions

    def save_model(self, path: str) -> None:
        """모델 저장"""
        self.model.save(path)
        logger.info(f"✓ 모델 저장 완료: {path}")

    def load_model(self, path: str) -> None:
        """모델 로드"""
        self.model.load(path)
        logger.info(f"✓ 모델 로드 완료: {path}")


class SegmentModelTrainer:
    """세그먼트별 모델 학습"""

    def __init__(self, lookback: int, save_dir: str):
        """
        Args:
            lookback: LSTM 입력 시퀀스 길이
            save_dir: 모델 저장 디렉토리
        """
        self.lookback = lookback
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}

    def train_segment_models(self, segment_data: Dict[str, pd.DataFrame],
                            feature_columns: List[str],
                            target_column: str,
                            train_end_date: str,
                            val_end_date: str,
                            **train_kwargs) -> Dict:
        """여러 세그먼트 모델 학습"""
        from src.models.lstm_model import LSTMModel, SequenceDataGenerator
        from src.preprocessor import DataNormalizer

        logger.info(f"\n세그먼트별 모델 학습 시작... ({len(segment_data)}개)")

        results = {}

        for segment_name, data in segment_data.items():
            try:
                logger.info(f"\n[{segment_name}] 학습 중...")

                # 데이터 분할
                train_data = data[data['date'] <= train_end_date]
                val_data = data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)]

                if len(train_data) < self.lookback + 1 or len(val_data) < 1:
                    logger.warning(f"[{segment_name}] 데이터 부족, 스킵")
                    continue

                # 특성과 타겟 추출
                X_train = train_data[feature_columns].values
                y_train = train_data[target_column].values

                X_val = val_data[feature_columns].values
                y_val = val_data[target_column].values

                # 정규화
                normalizer = DataNormalizer()
                normalizer.fit(train_data[[target_column]], [target_column])

                X_train_norm = normalizer.transform(train_data[feature_columns])
                X_val_norm = normalizer.transform(val_data[feature_columns])

                # 시퀀스 생성
                generator = SequenceDataGenerator(self.lookback)
                X_train_seq, y_train_seq = generator.create_sequences(
                    X_train_norm.values, y_train
                )
                X_val_seq, y_val_seq = generator.create_sequences(
                    X_val_norm.values, y_val
                )

                # 모델 구축 및 학습
                model = LSTMModel(self.lookback, X_train_seq.shape[2], segment_name)
                model.build_model()

                history = model.train(
                    X_train_seq, y_train_seq,
                    X_val_seq, y_val_seq,
                    **train_kwargs
                )

                # 모델 저장
                model_path = self.save_dir / f"{segment_name}.h5"
                model.save(str(model_path))

                self.models[segment_name] = model
                results[segment_name] = {
                    'status': 'success',
                    'samples': len(X_train_seq),
                }

                logger.info(f"✓ [{segment_name}] 학습 완료")

            except Exception as e:
                logger.warning(f"✗ [{segment_name}] 학습 실패: {str(e)}")
                results[segment_name] = {
                    'status': 'failed',
                    'error': str(e),
                }

        self.results = results
        return results

    def get_trained_models(self) -> Dict:
        """학습된 모델 반환"""
        return self.models

    def get_results(self) -> Dict:
        """학습 결과 반환"""
        return self.results


if __name__ == "__main__":
    print("Trainer 모듈 정의 완료")
