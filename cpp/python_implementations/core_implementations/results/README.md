# Results Directory

Generated results and performance data from experiments and tests.

## Files

- `bottleneck_analysis_*.json` - Bottleneck analysis results
- `large_scale_test_results_*.json` - Large-scale performance testing results
- `oscillation_demo_results_*.json` - Neural oscillation demonstration results
- `quantization_baseline_*.json` - Quantization baseline performance data
- `test_profile.json` - Performance profile data

## Data Format

JSON files with timestamp naming convention (`YYYYMMDD_HHMMSS`).

### Performance Results
```json
{
  "neuron_count": 1000000,
  "active_percentage": 0.01,
  "performance": {
    "steps_per_second": 418.5,
    "memory_usage_gb": 0.5,
    "gpu_utilization": 85.2
  }
}
```

### Neural Oscillation Results
```json
{
  "Deep Sleep": {
    "performance": {
      "total_neurons": 68000,
      "memory_requirements_gb": 0.008,
      "real_time_capable": true
    },
    "oscillations": {
      "Delta": {"frequency": 2.0, "neurons": 50000},
      "Theta": {"frequency": 6.0, "neurons": 10000}
    }
  }
}
```

## Usage

```python
import json
with open('results/large_scale_test_results_20250924_034444.json', 'r') as f:
    data = json.load(f)
```
