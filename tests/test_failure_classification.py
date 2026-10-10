"""Unit tests for failure classification engine (Phase 12)."""

from researchlens.execution.process_monitor import classify_failure


def test_failure_classification_categories():
    # 1. Timeout
    assert classify_failure(exit_code=-1, timed_out=True) == "TIMEOUT"

    # 2. Dependency Error
    dep_err = "Traceback:\nModuleNotFoundError: No module named 'torchvision'"
    assert classify_failure(exit_code=1, stderr=dep_err) == "DEPENDENCY_ERROR"

    import_err = "ImportError: cannot import name 'custom_layer'"
    assert classify_failure(exit_code=1, stderr=import_err) == "DEPENDENCY_ERROR"

    # 3. Configuration Error
    cfg_err = "error: unrecognized arguments: --bad_flag 123"
    assert classify_failure(exit_code=2, stderr=cfg_err) == "CONFIGURATION_ERROR"

    # 4. Data Error
    data_err = "FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'"
    assert classify_failure(exit_code=1, stderr=data_err) == "DATA_ERROR"

    # 5. Model Error
    model_err = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert classify_failure(exit_code=1, stderr=model_err) == "MODEL_ERROR"

    shape_err = "RuntimeError: shape mismatch: mat1 and mat2 shapes cannot be multiplied"
    assert classify_failure(exit_code=1, stderr=shape_err) == "MODEL_ERROR"

    # 6. Resource Error
    res_err = "MemoryError: Unable to allocate array with shape"
    assert classify_failure(exit_code=137, stderr=res_err) == "RESOURCE_ERROR"

    # 7. Unknown
    unknown_err = "An unknown failure occurred on line 42"
    assert classify_failure(exit_code=1, stderr=unknown_err) == "UNKNOWN"
