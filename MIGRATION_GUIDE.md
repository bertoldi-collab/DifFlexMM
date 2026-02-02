# Migration Guide: JAX/jaxlib Version Update

## Overview

This document explains the dependency updates made to fix CI/build failures and provides guidance for users migrating from the original version.

## What Changed

### Python Version
- **Before**: Python >= 3.8, < 3.12
- **After**: Python ^3.11 (Python 3.11 or 3.12)

### JAX/jaxlib Versions
- **Before**: JAX 0.4.8, jaxlib 0.4.7, jax-md 0.2.5
- **After**: JAX ^0.5.3, jaxlib ^0.5.3, jax-md ^0.2.26

### NumPy Version
- **Added Constraint**: numpy >= 1.21, < 2 (for compatibility)

## Why These Changes Were Necessary

### The Problem
1. **jaxlib 0.4.7 was removed from PyPI**: The original version specified in `poetry.lock` is no longer available for download from PyPI. This caused installation failures in CI and for new users.

2. **Version Compatibility Chain**: 
   - Available jaxlib versions start at 0.4.17+, which require Python >= 3.9
   - jax-md 0.2.5-0.2.7 are incompatible with JAX 0.4.17+ due to API changes (`jax.abstract_arrays` was moved)
   - jax-md 0.2.26+ requires jaxlib >= 0.5.0 and Python >= 3.11

3. **Ecosystem Evolution**: JAX has moved forward with breaking API changes that required updating the entire dependency stack.

## Data Compatibility

### Loading Published Data
**Good news**: Data saved with the original version can still be loaded with the new version.

The `load_data()` function in `difflexmm/utils.py` already includes logic to convert NumPy arrays to JAX arrays:

```python
def load_data(path_or_filename: Union[str, Path]):
    with open(path_or_filename, "rb") as file:
        data = pickle.load(file)
        
        if isinstance(data, (SolutionData, EigenmodeData)):
            # Cast arrays to jax arrays
            class_type = type(data)
            return class_type(*(jnp.array(attr) if isinstance(attr, np.ndarray) else attr for attr in data))
        
        return data
```

This ensures that:
- Data pickled with JAX 0.4.x arrays can be loaded
- Data pickled with NumPy arrays can be loaded
- The conversion happens transparently

### Saving New Data
Data saved with the new version will use the new JAX array format. If you need to ensure maximum compatibility with older versions, consider converting JAX arrays to NumPy arrays before pickling.

## Migration Steps

### For New Users
Simply install the package with Python 3.11:
```bash
pip install poetry
poetry install
```

### For Existing Users

1. **Update Python** to version 3.11 or later:
   ```bash
   # Using conda
   conda install python=3.11
   
   # Using pyenv
   pyenv install 3.11.x
   pyenv local 3.11.x
   ```

2. **Update Dependencies**:
   ```bash
   poetry install
   ```

3. **Test Your Code**:
   Run your existing tests to ensure compatibility:
   ```bash
   poetry run pytest tests/
   ```

### Known API Changes

Most code should work without changes. However, be aware of these JAX API changes:

1. **Random Key Generation**: If your code uses JAX random number generation, you may need to update to the new PRNG API introduced in JAX 0.4+.

2. **Deprecated Functions**: Some functions like `jax.tree_map` are deprecated in favor of `jax.tree.map` or `jax.tree_util.tree_map`.

## Troubleshooting

### Issue: Installation fails with Python 3.8-3.10
**Solution**: Upgrade to Python 3.11 or later. The newer dependencies require this Python version.

### Issue: Existing data files won't load
**Solution**: This shouldn't happen due to the backward-compatible `load_data()` function. If you encounter issues, please file a GitHub issue with:
- The error message
- The Python version used to save the data
- The Python version used to load the data

### Issue: Tests fail with deprecation warnings
**Solution**: These are usually harmless warnings from dependencies (especially jax-md). You can ignore them or filter them out:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

## CI/CD Changes

The GitHub Actions workflow now tests only Python 3.11:

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.11"]
```

If you need to test multiple Python versions, you can expand this matrix to include 3.12 or later versions (but not 3.8-3.10).

## References

- [JAX Changelog](https://jax.readthedocs.io/en/latest/changelog.html)
- [jax-md Repository](https://github.com/jax-md/jax-md)
- [Python EOL Dates](https://endoflife.date/python)

## Questions?

If you have questions about this migration, please:
1. Check the [GitHub Issues](https://github.com/bertoldi-collab/DifFlexMM/issues)
2. Review the [JAX Documentation](https://jax.readthedocs.io/)
3. Open a new issue if your question isn't addressed
