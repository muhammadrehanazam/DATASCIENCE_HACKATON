"""
Simple AI Fraud Detection System Demo
Demonstrates the 5 AI agents working together
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the AI system components
from ai_agents import (
    TransactionMonitor, PatternAnalyzer, BehaviorChecker, 
    CaseResearcher, AlertGenerator, AgentReport
)
from ai_orchestrator import AIOrchestrator

def create_sample_transaction():
    """Create a sample transaction for testing"""
    return {
        'transaction_id': f'TX_{np.random.randint(100000, 999999)}',
        'user_id': f'USER_{np.random.randint(1000, 9999)}',
        'amount': np.random.uniform(100, 5000),
        'timestamp': datetime.now(),
        'merchant': f'MERCHANT_{np.random.randint(100, 999)}',
        'merchant_category': np.random.choice(['retail', 'online', 'service', 'gambling']),
        'country': np.random.choice(['US', 'CA', 'GB', 'CN', 'RU']),
        'device_info': 'Chrome/Windows',
        'ip_address': f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}',
        'location': f'CITY_{np.random.randint(1, 100)}'
    }

def create_sample_user_history(user_id, num_transactions=20):
    """Create sample user history for testing"""
    dates = pd.date_range(end=datetime.now(), periods=num_transactions, freq='D')
    
    return pd.DataFrame({
        'transaction_id': [f'TX_{i}_{user_id}' for i in range(num_transactions)],
        'amount': np.random.lognormal(mean=4.0, sigma=1.0, size=num_transactions),
        'timestamp': dates,
        'merchant': np.random.choice(['MERCHANT_A', 'MERCHANT_B', 'MERCHANT_C'], num_transactions),
        'location': np.random.choice(['CITY_1', 'CITY_2', 'CITY_3'], num_transactions),
        'is_fraud': np.random.choice([False, False, False, True], num_transactions, p=[0.95, 0.05])
    })

def run_simple_demo():
    """Run a simple demonstration"""
    
    print("🎯 AI Fraud Detection System Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    print("🤖 Initializing AI agents...")
    orchestrator = AIOrchestrator()
    
    # Create sample data
    print("\n📋 Creating sample transaction...")
    transaction = create_sample_transaction()
    user_history = create_sample_user_history(transaction['user_id'])
    
    print(f"Transaction: ${transaction['amount']:.2f} by {transaction['user_id']}")
    print(f"User has {len(user_history)} historical transactions")
    
    # Run AI analysis
    print(f"\n🔍 Running AI analysis...")
    result = orchestrator.analyze_transaction(transaction, user_history)
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"Risk Score: {result['final_risk_score']:.2f}/1.0")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Alert Priority: {result['alert_priority']}")
    print(f"Processing Time: {result['orchestration_metadata']['total_processing_time']:.2f}s")
    
    print(f"\n🔍 Key Findings:")
    for finding in result['findings'][:3]:
        print(f"  • {finding}")
    
    print(f"\n💡 Recommendations:")
    for rec in result['recommendations'][:2]:
        print(f"  • {rec}")
    
    return result

def test_individual_agents():
    """Test each agent individually"""
    
    print("\n🧪 Testing Individual Agents")
    print("=" * 30)
    
    # Create sample data
    transaction = create_sample_transaction()
    user_history = create_sample_user_history(transaction['user_id'])
    
    # Test Transaction Monitor
    print("\n1️⃣ Transaction Monitor:")
    monitor = TransactionMonitor()
    report = monitor.analyze_transaction(transaction, user_history)
    print(f"   Risk Score: {report.risk_score:.2f}")
    print(f"   Status: {report.status}")
    
    # Test Pattern Analyzer
    print("\n2️⃣ Pattern Analyzer:")
    analyzer = PatternAnalyzer()
    engineered_features = {
        'credit_risk_score': 0.5,
        'velocity_24h': 0.3,
        'velocity_4w': 0.2,
        'email_device_risk': 0.1,
        'age_velocity_interaction': 0.4
    }
    report = analyzer.analyze_transaction(transaction, engineered_features)
    print(f"   Risk Score: {report.risk_score:.2f}")
    print(f"   Status: {report.status}")
    
    # Test Behavior Checker
    print("\n3️⃣ Behavior Checker:")
    checker = BehaviorChecker()
    report = checker.analyze_transaction(transaction, user_history)
    print(f"   Risk Score: {report.risk_score:.2f}")
    print(f"   Status: {report.status}")
    
    # Test Case Researcher
    print("\n4️⃣ Case Researcher:")
    researcher = CaseResearcher()
    report = researcher.analyze_transaction(transaction, user_history)
    print(f"   Risk Score: {report.risk_score:.2f}")
    print(f"   Status: {report.status}")

if __name__ == "__main__":
    # Run main demo
    result = run_simple_demo()
    
    # Test individual agents
    test_individual_agents()
    
    print("\n✅ AI System Demo Completed!")
    print("The AI agents are working together to detect fraud!")