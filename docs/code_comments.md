# Code Commenting & Documentation

## Guidelines

- Use docstrings for all functions, classes, and modules.
- Inline comments for complex logic.
- Use type hints where possible.

## Example

```python
def predict_event(input_features: dict) -> float:
    """
    Run prediction on input features using the pre-trained model.

    Args:
        input_features (dict): A dictionary of feature values.

    Returns:
        float: Predicted event probability.
    """
    # Prepare features
    features = preprocess(input_features)
    # Run model
    return model.predict(features)
```

## Tools

- Use [pdoc](https://pdoc.dev/) or [Sphinx](https://www.sphinx-doc.org/) for auto-generating documentation.

---

**Maintain comments as code evolves.**