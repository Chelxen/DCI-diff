"""Conversions shared by model output, metrics, and CT data consistency."""


def to_raw_intensity(value, data_range=1.0, raw_data=False):
    """Raw mode is an identity; legacy mode decodes [-1, 1]."""
    return value if raw_data else (value + 1.0) * (data_range / 2.0)


def from_raw_intensity(value, data_range=1.0, raw_data=False):
    return value if raw_data else value * (2.0 / data_range) - 1.0


def validate_intensity_config(model_opt, dataset_opt):
    """Prevent mixing raw GE data with a normalized model configuration."""
    raw_data = bool(model_opt.get('raw_data', False))
    if raw_data != bool(dataset_opt.get('raw_data', False)):
        raise ValueError('raw_data must match in the model JSON and dataset YAML')
    if float(model_opt.get('data_range', 1.0)) <= 0:
        raise ValueError('data_range must be positive')
    return raw_data
