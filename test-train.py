def test_training_model():
    from notebooks.train import train_and_log_model
    try:
        train_and_log_model()
    except Exception as e:
        assert False, f"Model training or registration failed with error: {e}"

    assert True  # Training and registration successful
