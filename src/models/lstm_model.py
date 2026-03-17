"""
LSTM 모델 정의
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM 시계열 예측 모델"""

    def __init__(self, lookback: int, n_features: int, model_name: str = "lstm"):
        """
        Args:
            lookback: 입력 시퀀스 길이
            n_features: 입력 특성 수
            model_name: 모델명
        """
        self.lookback = lookback
        self.n_features = n_features
        self.model_name = model_name
        self.model = None
        self.history = None

    def build_model(self, lstm_units: list = None, dropout: float = 0.2,
                    bidirectional: bool = False, lr_warmup_epochs: int = 5) -> keras.Model:
        """모델 구축

        Args:
            lstm_units: 각 LSTM 레이어의 유닛 수 리스트
            dropout: 드롭아웃 비율
            bidirectional: True이면 Bidirectional LSTM 사용, False이면 단방향 LSTM 사용
            lr_warmup_epochs: LR Warmup 스케줄러에서 사용할 워밍업 에포크 수 (저장용)
        """
        if lstm_units is None:
            lstm_units = [128, 64, 32]

        lstm_type = "Bidirectional LSTM" if bidirectional else "Unidirectional LSTM"
        logger.info(f"모델 구축 중... ({self.model_name})")
        logger.info(f"- Input shape: ({self.lookback}, {self.n_features})")
        logger.info(f"- LSTM units: {lstm_units}")
        logger.info(f"- LSTM 유형: {lstm_type}")

        self._lr_warmup_epochs = lr_warmup_epochs

        def make_lstm_layer(units, return_sequences, is_first=False):
            """단방향 또는 양방향 LSTM 레이어 생성"""
            if is_first:
                lstm = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    input_shape=(self.lookback, self.n_features)
                )
            else:
                lstm = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout
                )
            if bidirectional:
                return layers.Bidirectional(lstm)
            return lstm

        model = Sequential([
            # LSTM 레이어 1
            make_lstm_layer(lstm_units[0], return_sequences=True, is_first=True),

            # LSTM 레이어 2
            make_lstm_layer(lstm_units[1], return_sequences=True),

            # LSTM 레이어 3
            make_lstm_layer(lstm_units[2], return_sequences=False),

            # Dense 레이어
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),

            layers.Dense(16, activation='relu'),
            layers.Dropout(dropout),

            # 출력 레이어
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"✓ 모델 구축 완료 ({lstm_type})")
        model.summary()

        self.model = model
        return model

    def _get_lr_scheduler(self, warmup_epochs: int = 5, base_lr: float = 0.001):
        """LR Warmup 스케줄러 반환

        Args:
            warmup_epochs: 워밍업 에포크 수
            base_lr: 기본 학습률

        Returns:
            LearningRateScheduler 콜백
        """
        def scheduler(epoch, lr):
            if epoch < warmup_epochs:
                return base_lr * (epoch + 1) / warmup_epochs
            return lr

        logger.info(f"- LR Warmup 스케줄러 활성화 (워밍업 에포크: {warmup_epochs}, 기본 LR: {base_lr})")
        return tf.keras.callbacks.LearningRateScheduler(scheduler)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 150, batch_size: int = 32,
              early_stopping_patience: int = 15,
              use_lr_warmup: bool = False) -> keras.callbacks.History:
        """모델 학습

        Args:
            X_train: 학습 입력 데이터
            y_train: 학습 타겟 데이터
            X_val: 검증 입력 데이터
            y_val: 검증 타겟 데이터
            epochs: 최대 에포크 수
            batch_size: 배치 크기
            early_stopping_patience: EarlyStopping 인내 에포크 수
            use_lr_warmup: True이면 LR Warmup 스케줄러를 콜백에 추가
        """
        logger.info(f"\n모델 학습 시작...")
        logger.info(f"- Train samples: {X_train.shape[0]}")
        logger.info(f"- Val samples: {X_val.shape[0]}")
        logger.info(f"- Epochs: {epochs}, Batch size: {batch_size}")

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [early_stop, reduce_lr]

        if use_lr_warmup:
            warmup_epochs = getattr(self, '_lr_warmup_epochs', 5)
            lr_scheduler = self._get_lr_scheduler(warmup_epochs=warmup_epochs)
            callbacks.append(lr_scheduler)
            logger.info("- LR Warmup 스케줄러가 콜백에 추가되었습니다.")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("✓ 학습 완료")
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("모델이 구축되지 않았습니다.")

        predictions = self.model.predict(X, verbose=0)
        return predictions

    def save(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        self.model.save(path)
        logger.info(f"✓ 모델 저장: {path}")

    def load(self, path: str) -> None:
        """모델 로드"""
        self.model = keras.models.load_model(path)
        logger.info(f"✓ 모델 로드: {path}")

    def get_history(self) -> dict:
        """학습 이력 반환"""
        if self.history is None:
            return None

        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', []),
            'lr': self.history.history.get('lr', []),
        }


class SequenceDataGenerator:
    """시퀀스 데이터 생성"""

    def __init__(self, lookback: int):
        """
        Args:
            lookback: 입력 시퀀스 길이
        """
        self.lookback = lookback

    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> tuple:
        """
        시퀀스 데이터 생성

        Args:
            data: 입력 데이터 (n_samples, n_features)
            target: 타겟 데이터 (n_samples,). None이면 data의 첫 컬럼 사용

        Returns:
            X: (n_sequences, lookback, n_features)
            y: (n_sequences,)
        """
        if target is None:
            target = data[:, 0]

        X, y = [], []

        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(target[i+self.lookback])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"✓ 시퀀스 생성 완료: X={X.shape}, y={y.shape}")

        return X, y

    def create_sequences_with_features(self, data: np.ndarray,
                                      feature_indices: list = None) -> tuple:
        """
        특정 특성만 사용하여 시퀀스 생성

        Args:
            data: 입력 데이터 (n_samples, n_features)
            feature_indices: 사용할 특성 인덱스. None이면 모두 사용

        Returns:
            X: (n_sequences, lookback, n_selected_features)
            y: (n_sequences,)
        """
        if feature_indices is None:
            feature_indices = list(range(data.shape[1]))

        target_idx = feature_indices[0]  # 첫 번째 특성을 타겟으로

        X, y = [], []

        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback][:, feature_indices])
            y.append(data[i+self.lookback, target_idx])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"✓ 시퀀스 생성 완료: X={X.shape}, y={y.shape}")

        return X, y


if __name__ == "__main__":
    # 테스트
    from config import LOOKBACK_WINDOW, LSTM_PARAMS

    # 더미 데이터
    X_dummy = np.random.rand(1000, LOOKBACK_WINDOW, 25)
    y_dummy = np.random.rand(1000)

    X_train = X_dummy[:700]
    y_train = y_dummy[:700]
    X_val = X_dummy[700:850]
    y_val = y_dummy[700:850]

    # 모델 생성 및 학습
    lstm = LSTMModel(LOOKBACK_WINDOW, 25)
    lstm.build_model()

    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,  # 테스트용 짧은 에포크
        batch_size=32
    )

    # 예측
    predictions = lstm.predict(X_val[:5])
    print(f"\n테스트 예측 (처음 5개):\n{predictions.flatten()}")
