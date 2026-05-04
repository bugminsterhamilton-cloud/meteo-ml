"""Model registry for forecast service."""
from .base import ForecastModel
from .stats import StatsForecastModel
from .tabular import TabularForecastModel

# Remove SequenceForecastModel from startup imports to avoid TensorFlow loading delays
# Uncomment if you want to use neural sequence models:
# from .sequence import SequenceForecastModel

MODEL_REGISTRY = {
    "tabular": TabularForecastModel,
    "stats": StatsForecastModel,
    # "sequence": SequenceForecastModel,  # Disabled due to TensorFlow startup overhead
}

__all__ = ["ForecastModel", "StatsForecastModel", "TabularForecastModel", "MODEL_REGISTRY"]
