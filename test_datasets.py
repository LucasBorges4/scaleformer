#!/usr/bin/env python3
"""
Test script to verify Scaleformer dataset functionality
"""

import pandas as pd
import numpy as np
import tempfile
import os
from data_provider.data_factory import data_provider
from data_provider.dataset_registry import DatasetRegistry, create_dataset

class Args:
    """Mock args object for testing"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_registry_system():
    """Test dataset registry system directly"""
    print("Testing dataset registry system...")
    
    try:
        # List available dataset types
        available = DatasetRegistry.list_types()
        print(f"✓ Available dataset types: {available}")
        
        # Test CSV dataset creation directly
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV file
            csv_file = os.path.join(temp_dir, 'ETTh1.csv')
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='h'),
                'value': np.random.randn(100).cumsum()
            })
            data.to_csv(csv_file, index=False)
            
            try:
                dataset = create_dataset(
                    data_source='ETTh1',  # Use registered name
                    root_path=temp_dir,
                    data_path='ETTh1.csv',  # File name
                    flag='train',
                    size=[96, 48, 24],
                    features='M',
                    target='value',
                    freq='h'
                )
                
                print(f"✓ CSV dataset created via registry")
                print(f"  - Length: {len(dataset)}")
                print(f"  - Data shape: {dataset.data_x.shape}")
                
            except Exception as e:
                print(f"✗ CSV dataset creation failed: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Registry system test failed: {e}")

def test_data_provider_with_args():
    """Test data_provider function with proper args object"""
    print("\nTesting data_provider function...")
    
    try:
        # Create temporary directory with CSV file
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'ETTh1.csv')
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=200, freq='h'),
                'value': np.random.randn(200).cumsum()
            })
            data.to_csv(csv_file, index=False)
            
            # Create mock args object
            args = Args(
                data_source='ETTh1',  # Use registered name
                root_path=temp_dir,
                data_path='ETTh1.csv',
                features='M',
                target='value',
                freq='h',
                seq_len=96,
                label_len=48,
                pred_len=24,
                embed='',
                scale=True
            )
            
            # Test with train flag
            dataset, data_loader = data_provider(args, 'train')
            print(f"✓ data_provider created dataset successfully")
            print(f"  - Dataset length: {len(dataset)}")
            print(f"  - DataLoader created")
            
    except Exception as e:
        print(f"✗ data_provider test failed: {e}")
        import traceback
        traceback.print_exc()

def test_legacy_datasets():
    """Test legacy dataset types"""
    print("\nTesting legacy datasets...")
    
    try:
        # Test synthetic dataset
        args = Args(
            data='synthetic',  # Legacy parameter
            root_path='any',  # Doesn't matter for synthetic
            features='M',
            target='value',
            freq='h',
            seq_len=96,
            label_len=48,
            pred_len=24,
            embed='',
            scale=True
        )
        
        dataset, data_loader = data_provider(args, 'train')
        print(f"✓ Synthetic dataset works")
        print(f"  - Length: {len(dataset)}")
        print(f"  - Data shape: {dataset.data_x.shape}")
        
    except Exception as e:
        print(f"✗ Synthetic dataset failed: {e}")
        import traceback
        traceback.print_exc()

def test_sql_dataset():
    """Test SQL dataset functionality"""
    print("\nTesting SQL dataset...")
    
    try:
        # Create sample data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='h'),
            'value': np.random.randn(100).cumsum()
        })
        
        # Save to temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            data.to_sql('time_series', conn, index=False, if_exists='replace')
            conn.close()
            
            # Test dataset creation via registry
            args = Args(
                data_source='sqlite',
                root_path=f"sqlite:///{db_path}",
                data_path='time_series',
                features='M',
                target='value',
                freq='h',
                seq_len=96,
                label_len=48,
                pred_len=24,
                embed='',
                scale=True
            )
            
            dataset, data_loader = data_provider(args, 'train')
            print(f"✓ SQL dataset created successfully")
            print(f"  - Length: {len(dataset)}")
            print(f"  - Data shape: {dataset.data_x.shape}")
            
        except ImportError:
            print("⚠ SQL dataset skipped (sqlite3 not available)")
        except Exception as e:
            print(f"✗ SQL dataset failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    except Exception as e:
        print(f"✗ SQL test setup failed: {e}")

if __name__ == "__main__":
    print("Scaleformer Dataset Test Suite")
    print("=" * 40)
    
    test_registry_system()
    test_data_provider_with_args()
    test_legacy_datasets()
    test_sql_dataset()
    
    print("\n" + "=" * 40)
    print("Test suite completed!")