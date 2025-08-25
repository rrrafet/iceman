#!/usr/bin/env python3
"""
Test script to demonstrate risk analysis integration workflow.

This script simulates the notebook workflow and tests the integration
between risk analysis components and Streamlit.
"""

import sys
import os
import pickle
from datetime import datetime

# Add Spark modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from services.risk_service import RiskAnalysisService
    from data_loader import DataLoader
    print("✅ Maverick modules loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load Maverick modules: {e}")
    sys.exit(1)

def test_mock_data_generation():
    """Test mock data generation and caching"""
    print("\n" + "="*50)
    print("TEST 1: Mock Data Generation and Caching")
    print("="*50)
    
    # Initialize RiskAnalysisService without portfolio components
    service = RiskAnalysisService()
    
    # Generate mock data
    mock_data = service.get_risk_data("TOTAL")
    print(f"✅ Generated mock data with keys: {list(mock_data.keys())}")
    
    # Check cache info
    cache_info = service.get_cache_info()
    print(f"✅ Cache directory: {cache_info['cache_dir']}")
    print(f"✅ Cache files: {len(cache_info['cache_files'])}")
    
    return mock_data

def test_data_loader_integration():
    """Test DataLoader integration with RiskAnalysisService"""
    print("\n" + "="*50)
    print("TEST 2: DataLoader Integration")
    print("="*50)
    
    # Test DataLoader with RiskAnalysisService
    data_loader = DataLoader(use_risk_service=True)
    
    print(f"✅ DataLoader initialized")
    print(f"✅ Has risk service: {data_loader.has_risk_service()}")
    print(f"✅ Data keys: {list(data_loader.data.keys())}")
    
    # Test refresh functionality
    try:
        refreshed_data = data_loader.refresh_data("TOTAL")
        print(f"✅ Data refresh successful")
        print(f"✅ Refreshed data timestamp: {refreshed_data.get('metadata', {}).get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Data refresh failed: {e}")
    
    return data_loader

def test_cache_operations():
    """Test cache save/load operations"""
    print("\n" + "="*50)
    print("TEST 3: Cache Operations")
    print("="*50)
    
    service = RiskAnalysisService()
    
    # Create a mock schema-like object
    mock_schema = type('MockSchema', (), {
        'data': {
            'metadata': {
                'analysis_type': 'test',
                'timestamp': datetime.now().isoformat(),
                'source': 'test_integration.py'
            },
            'test_section': {
                'test_metric': 42.0
            }
        }
    })()
    
    # Save to cache
    cache_path = service.save_schema_to_cache(mock_schema, "TEST")
    if cache_path:
        print(f"✅ Schema saved to: {cache_path}")
        
        # Load from cache
        loaded_schema = service.load_schema_from_cache("TEST")
        if loaded_schema:
            print(f"✅ Schema loaded from cache")
            print(f"✅ Loaded data keys: {list(loaded_schema.data.keys())}")
        else:
            print("❌ Failed to load schema from cache")
    else:
        print("❌ Failed to save schema to cache")

def test_serialization():
    """Test numpy array serialization"""
    print("\n" + "="*50)
    print("TEST 4: Data Serialization")
    print("="*50)
    
    try:
        import numpy as np
        
        # Create data with numpy arrays
        test_data = {
            'weights': {
                'portfolio_weights': {'A': np.float64(0.5), 'B': np.float64(0.5)},
                'arrays': np.array([0.5, 0.5])
            },
            'metrics': {
                'risk_value': np.float64(0.025),
                'factor_exposures': np.array([0.1, 0.2, 0.3])
            }
        }
        
        service = RiskAnalysisService()
        serialized = service._serialize_schema_data(test_data)
        
        print(f"✅ Serialization successful")
        print(f"✅ Portfolio weights: {serialized['weights']['portfolio_weights']}")
        print(f"✅ Arrays converted: {type(serialized['weights']['arrays'])}")
        print(f"✅ Risk value converted: {type(serialized['metrics']['risk_value'])}")
        
    except ImportError:
        print("⚠️ NumPy not available, skipping serialization test")
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")

def simulate_notebook_workflow():
    """Simulate the typical notebook → Streamlit workflow"""
    print("\n" + "="*50)
    print("WORKFLOW: Notebook → Streamlit Integration")
    print("="*50)
    
    print("📓 Step 1: Simulate notebook risk analysis...")
    # This would be your actual risk analysis in the notebook
    mock_analysis_result = {
        'metadata': {
            'analysis_type': 'hierarchical',
            'timestamp': datetime.now().isoformat(),
            'component': 'TOTAL',
            'source': 'notebook_simulation'
        },
        'portfolio': {
            'core_metrics': {
                'TOTAL': {
                    'total_risk': 0.0245,
                    'factor_risk_contribution': 0.0175,
                    'specific_risk_contribution': 0.0070
                }
            }
        },
        'active': {
            'core_metrics': {
                'TOTAL': {
                    'total_risk': 0.0085,
                    'factor_risk_contribution': 0.0055,
                    'specific_risk_contribution': 0.0030
                }
            }
        },
        'weights': {
            'portfolio_weights': {'CA': -0.165, 'IG': 0.381, 'EQLIKE': 0.784},
            'benchmark_weights': {'CA': -0.158, 'IG': 0.381, 'EQLIKE': 0.777},
            'active_weights': {'CA': -0.007, 'IG': 0.000, 'EQLIKE': 0.007}
        }
    }
    
    print("💾 Step 2: Save to Maverick cache...")
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    cache_file = os.path.join(cache_dir, 'risk_data_TOTAL.pkl')
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(mock_analysis_result, f)
        print(f"✅ Analysis saved to: {cache_file}")
    except Exception as e:
        print(f"❌ Failed to save analysis: {e}")
        return
    
    print("🔄 Step 3: Test Streamlit data loading...")
    try:
        # Simulate what happens when Streamlit loads the cached data
        data_loader = DataLoader(use_risk_service=True)
        loaded_data = data_loader.data
        
        print(f"✅ Streamlit loaded data with keys: {list(loaded_data.keys())}")
        
        # Verify key metrics
        portfolio_metrics = loaded_data.get('portfolio', {}).get('core_metrics', {}).get('TOTAL', {})
        if portfolio_metrics:
            print(f"✅ Portfolio total risk: {portfolio_metrics.get('total_risk', 'N/A')}")
        
        weights = loaded_data.get('weights', {})
        if weights:
            print(f"✅ Portfolio weights available: {len(weights.get('portfolio_weights', {}))}")
        
    except Exception as e:
        print(f"❌ Streamlit loading failed: {e}")

def main():
    """Run all integration tests"""
    print("🚀 Maverick Risk Analysis Integration Tests")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Run tests
        test_mock_data_generation()
        test_data_loader_integration() 
        test_cache_operations()
        test_serialization()
        simulate_notebook_workflow()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS COMPLETED")
        print("="*50)
        print("""
📋 Next Steps for Notebook Integration:

1. In your Jupyter notebook, after running risk analysis:
   ```python
   # Save your schema to Maverick cache
   import sys
   sys.path.append('/Users/rafet/Workspace/Spark/spark-ui/apps/maverick')
   from services.risk_service import RiskAnalysisService
   
   service = RiskAnalysisService()
   service.save_schema_to_cache(schema, "TOTAL")
   ```

2. In Streamlit app:
   - Click "🔄 Refresh Risk Data" button
   - Or navigate to "Data Management" tab
   - Check cache files and refresh status

3. For real-time integration:
   - Initialize DataLoader with portfolio_graph and factor_returns
   - Enable direct risk analysis generation
        """)
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()