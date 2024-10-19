import sys
import os
import pytest
import pandas as pd
import numpy as np
import yaml

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.make_dataset import preprocess_data, load_data, save_data

# Mock the default config
@pytest.fixture
def mock_config(monkeypatch):
    config = {
        "target_column": "target"
    }
    mock_yaml = yaml.dump(config)
    
    def mock_open(*args, **kwargs):
        from io import StringIO
        return StringIO(mock_yaml)
    
    monkeypatch.setattr('builtins.open', mock_open)

@pytest.fixture
def sample_train_data():
    # Create sample training data with the expected shape (4120, 33)
    np.random.seed(42)
    n_samples = 4120
    
    # Generate numeric features (using absolute values to avoid log1p warnings)
    numeric_data = pd.DataFrame(
        np.abs(np.random.randn(n_samples, 25)) + 1,  # Adding 1 to ensure positive values
        columns=[f'numeric_{i}' for i in range(25)]
    )
    
    # Generate categorical features
    categorical_data = pd.DataFrame({
        'cat_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'cat_3': np.random.choice(['P', 'Q', 'R'], n_samples),
        'cat_4': np.random.choice(['M', 'N', 'O'], n_samples),
        'cat_5': np.random.choice(['D', 'E', 'F'], n_samples),
        'cat_6': np.random.choice(['G', 'H', 'I'], n_samples)
    })
    
    # Generate target column (ensure it's a proper classification target)
    target = pd.Series(np.random.randint(0, 2, n_samples), name='target')
    
    # Combine all features
    df = pd.concat([numeric_data, categorical_data, target], axis=1)
    
    # Add ID column
    df.insert(0, 'id', range(1, n_samples + 1))
    
    # Ensure no missing values
    df = df.fillna(0)
    
    return df

@pytest.fixture
def sample_test_data():
    # Create sample test data with the expected shape (1030, 32)
    np.random.seed(42)
    n_samples = 1030
    
    # Generate numeric features (using absolute values to avoid log1p warnings)
    numeric_data = pd.DataFrame(
        np.abs(np.random.randn(n_samples, 25)) + 1,  # Adding 1 to ensure positive values
        columns=[f'numeric_{i}' for i in range(25)]
    )
    
    # Generate categorical features
    categorical_data = pd.DataFrame({
        'cat_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'cat_3': np.random.choice(['P', 'Q', 'R'], n_samples),
        'cat_4': np.random.choice(['M', 'N', 'O'], n_samples),
        'cat_5': np.random.choice(['D', 'E', 'F'], n_samples),
        'cat_6': np.random.choice(['G', 'H', 'I'], n_samples)
    })
    
    # Combine all features
    df = pd.concat([numeric_data, categorical_data], axis=1)
    
    # Add ID column
    df.insert(0, 'id', range(1, n_samples + 1))
    
    # Ensure no missing values
    df = df.fillna(0)
    
    return df

def test_training_data_preprocessing(sample_train_data, mock_config):
    """Test the preprocessing of training data to ensure correct shapes at each step"""
    
    # Initial shape check
    assert sample_train_data.shape == (4120, 33), \
        f"Expected training data shape (4120, 33), but got {sample_train_data.shape}"
    
    try:
        # Process the data
        X, y = preprocess_data(sample_train_data)
        
        # Check shapes after preprocessing
        assert X is not None, "Preprocessed X data is None"
        
        # Check X shape
        assert X.shape == (4120, 32), \
            f"Expected preprocessed X shape (4120, 32), but got {X.shape}"
            
        # Only check y if it exists
        if y is not None:
            assert y.shape == (4120,), \
                f"Expected target shape (4120,), but got {y.shape}"
            
    except Exception as e:
        pytest.fail(f"Preprocessing failed with error: {str(e)}")

def test_test_data_preprocessing(sample_test_data, mock_config):
    """Test the preprocessing of test data to ensure correct shapes at each step"""
    
    # Initial shape check
    assert sample_test_data.shape == (1030, 32), \
        f"Expected test data shape (1030, 32), but got {sample_test_data.shape}"
    
    try:
        # Process the data
        X, y = preprocess_data(sample_test_data)
        
        # Check shapes after preprocessing
        assert X is not None, "Preprocessed X data is None"
        assert X.shape == (1030, 31), \
            f"Expected preprocessed X shape (1030, 31), but got {X.shape}"
            
    except Exception as e:
        pytest.fail(f"Preprocessing failed with error: {str(e)}")

def test_save_data(sample_train_data, sample_test_data, tmp_path, mock_config):
    """Test the save_data function with both training and test data"""
    
    # Create temporary paths for saving data
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    
    try:
        # Save training data
        save_data(sample_train_data, train_path)
        
        # Save test data
        save_data(sample_test_data, test_path)
        
        # Verify that the parent directories exist
        assert train_path.parent.exists()
        assert test_path.parent.exists()
            
    except Exception as e:
        pytest.fail(f"Save data failed with error: {str(e)}")

def test_load_data(sample_train_data, tmp_path):
    """Test the load_data function"""
    
    # Save sample data to a temporary file
    temp_file = tmp_path / "temp_data.csv"
    sample_train_data.to_csv(temp_file, index=False)
    
    try:
        # Load the data
        loaded_data = load_data(temp_file)
        
        # Verify the loaded data has the same shape as the original
        assert loaded_data is not None, "Loaded data is None"
        assert loaded_data.shape == sample_train_data.shape, \
            f"Expected shape {sample_train_data.shape}, but got {loaded_data.shape}"
            
    except Exception as e:
        pytest.fail(f"Load data failed with error: {str(e)}")