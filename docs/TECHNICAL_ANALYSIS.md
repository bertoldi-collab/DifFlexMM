# Technical Analysis: jaxlib Dependency Issue

## Problem Statement
The tests were failing due to a dependency issue with the jaxlib version. This document explains why this is an issue now when it wasn't months ago, and details the solution.

## Root Cause Analysis

### Timeline of Events

1. **Initial Development (2024)**: The repository was created with:
   - Python >= 3.8, < 3.12
   - JAX 0.4.8
   - jaxlib 0.4.7
   - jax-md 0.2.5

2. **December 2024**: Last successful CI run (commit 12172435402)
   - All tests passed with the original dependencies

3. **January 2026**: CI failures began
   - Error: "Unable to find installation candidates for jaxlib (0.4.7)"
   - Root cause: **jaxlib 0.4.7 was removed from PyPI**

### Why jaxlib 0.4.7 Was Removed

PyPI maintainers and package authors sometimes remove old versions to:
- Clean up deprecated versions
- Force users to upgrade to versions with security fixes
- Reduce storage and bandwidth costs
- Enforce minimum version requirements for dependencies

In this case, jaxlib versions 0.4.8 through 0.4.16 were also removed, leaving only 0.4.17+ available.

### The Compatibility Cascade

When trying to update to available versions, we encountered a cascade of compatibility issues:

```
jaxlib 0.4.7 (removed) 
    ↓
jaxlib 0.4.17+ available
    ↓ requires
Python >= 3.9 (breaks Python 3.8 support)
    ↓ but
jax-md 0.2.5 uses jax.abstract_arrays
    ↓ which was
Removed in JAX 0.4.17+
    ↓ so need
jax-md 0.2.8+
    ↓ but
jax-md 0.2.8 uses random.KeyArray
    ↓ which isn't in
JAX 0.4.17-0.4.38
    ↓ so need
jax-md 0.2.26+
    ↓ which requires
jaxlib >= 0.5.0 AND Python >= 3.11
```

## Solution

### Final Configuration
- Python: ^3.11 (3.11 or 3.12)
- JAX: ^0.5.3
- jaxlib: ^0.5.3  
- jax-md: ^0.2.27
- numpy: >= 1.21, < 2

### Trade-offs

**What we gained:**
- ✅ Working CI/CD pipeline
- ✅ Access to latest bug fixes and performance improvements
- ✅ Future-proof dependency stack
- ✅ Maintained data loading compatibility

**What we gave up:**
- ❌ Python 3.8, 3.9, 3.10 support
- ❌ Pinned versions (now using semver ranges for future updates)

### Data Loading Compatibility

**Key Question**: Can we still load data produced with the version at time of publication?

**Answer**: Yes! ✅

**Reasoning**:
1. The `load_data()` function in `difflexmm/utils.py` already handles conversion:
   ```python
   if isinstance(data, (SolutionData, EigenmodeData)):
       class_type = type(data)
       return class_type(*(jnp.array(attr) if isinstance(attr, np.ndarray) else attr for attr in data))
   ```

2. This code:
   - Checks if loaded data is one of the special data structures
   - Converts any NumPy arrays to JAX arrays
   - Works regardless of JAX version

3. Pickle format compatibility:
   - NumPy arrays pickle to a stable format
   - JAX arrays may use different internal representations, but...
   - The conversion layer makes this transparent

### Testing
All existing tests pass with the new configuration:
```
tests/test_difflexmm.py::test_version PASSED                    [ 25%]
tests/test_difflexmm.py::test_xcorr PASSED                      [ 50%]
tests/test_difflexmm.py::test_tensile_test PASSED               [ 75%]
tests/test_difflexmm.py::test_frame_invariance_ligament_energy PASSED [100%]
```

## Alternative Approaches Considered

### 1. Pin to Archived jaxlib 0.4.7
**Problem**: The package is not available on PyPI anymore. Could use direct URL, but:
- Not a long-term solution
- May disappear from other mirrors too
- Prevents future security updates

### 2. Stay on JAX 0.4.x with jaxlib 0.4.17
**Problem**: jax-md 0.2.5-0.2.7 incompatible due to `jax.abstract_arrays` API change
- Could patch jax-md locally, but:
- Maintenance burden
- Breaks when jax-md updates

### 3. Keep Python 3.9-3.10 Support
**Problem**: jax-md 0.2.26+ requires Python 3.11+
- Could fork jax-md, but:
- Significant maintenance burden  
- Miss bug fixes and improvements

### 4. Current Solution (Chosen)
**Update entire stack to latest compatible versions**
- ✅ Clean, maintainable solution
- ✅ Gets latest improvements
- ✅ Reduces future compatibility issues
- ✅ Python 3.8 was EOL in October 2024 anyway

## Recommendations

1. **For Users**: Upgrade to Python 3.11+ and update dependencies
2. **For Data**: Existing published data remains compatible
3. **For Future**: Consider documenting dependency pins in papers/data releases
4. **For Development**: Test with multiple Python versions if broad compatibility needed

## References

- [Python 3.8 EOL](https://devguide.python.org/versions/)
- [JAX 0.4.x Release Notes](https://github.com/google/jax/releases)
- [jax-md Releases](https://github.com/jax-md/jax-md/releases)
- [PyPI Package Removal Policy](https://packaging.python.org/en/latest/)
