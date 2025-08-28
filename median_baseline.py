def median_baseline_submission():
    """Simple median baseline for comparison"""
    
    # Load any cached features to get test session IDs
    cache_dir = Path("data/features")
    with open(cache_dir / 'research_features.pkl', 'rb') as f:
        train_fs, val_fs, test_fs = pickle.load(f)
    
    # Combine all training data
    all_train = pd.concat([train_fs, val_fs], ignore_index=True)
    
    # Use median session value
    median_value = all_train['session_value'].median()
    
    print(f"Median baseline value: {median_value:.2f}")
    
    # Create submission
    submission = pd.DataFrame({
        'user_session': test_fs['user_session'],
        'session_value': median_value
    })
    
    submission.to_csv('median_baseline.csv', index=False)
    print("Median baseline saved!")
    
    return submission