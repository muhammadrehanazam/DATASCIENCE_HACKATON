"""
Simple AI Fraud Detection System
5 Specialized AI Agents working together
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AgentReport:
    """Standardized report format for all AI agents"""
    agent_name: str
    risk_score: float  # 0.0 to 1.0
    confidence: float    # 0.0 to 1.0
    findings: List[str]
    recommendations: List[str]
    processing_time: float
    status: str  # 'SUCCESS', 'ERROR', 'WARNING'

class TransactionMonitor:
    """Checks amount and frequency patterns"""
    
    def __init__(self):
        self.name = "TransactionMonitor"
        self.thresholds = {
            'high_amount': 5000,
            'unusual_frequency': 5,
            'rapid_sequence': 300
        }
    
    def analyze_transaction(self, transaction: Dict, user_history: pd.DataFrame) -> AgentReport:
        """Analyze transaction for amount and frequency anomalies"""
        
        start_time = datetime.now()
        findings = []
        risk_score = 0.0
        
        try:
            amount = transaction.get('amount', 0)
            
            # Check high amount
            if amount > self.thresholds['high_amount']:
                findings.append(f"High transaction amount: ${amount:.2f}")
                risk_score += 0.3
            
            # Check frequency patterns
            if len(user_history) > 0:
                recent_transactions = user_history[
                    user_history['timestamp'] > datetime.now() - timedelta(hours=1)
                ]
                
                if len(recent_transactions) >= self.thresholds['unusual_frequency']:
                    findings.append(f"Unusual frequency: {len(recent_transactions)} transactions in last hour")
                    risk_score += 0.4
            
            risk_score = min(risk_score, 1.0)
            
            return AgentReport(
                agent_name=self.name,
                risk_score=risk_score,
                confidence=0.8,
                findings=findings,
                recommendations=["Flag for review if risk > 0.5"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='SUCCESS'
            )
            
        except Exception as e:
            return AgentReport(
                agent_name=self.name,
                risk_score=0.0,
                confidence=0.0,
                findings=[f"Error: {str(e)}"],
                recommendations=["Manual review required"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='ERROR'
            )

class PatternAnalyzer:
    """Uses ML models to detect fraud patterns"""
    
    def __init__(self):
        self.name = "PatternAnalyzer"
        self.risk_weights = {
            'credit_risk_score': 0.3,
            'velocity_24h': 0.25,
            'velocity_4w': 0.2,
            'email_device_risk': 0.15,
            'age_velocity_interaction': 0.1
        }
    
    def analyze_transaction(self, transaction: Dict, engineered_features: Dict) -> AgentReport:
        """Analyze transaction using ML pattern detection"""
        
        start_time = datetime.now()
        findings = []
        risk_score = 0.0
        
        try:
            # Calculate weighted risk score from engineered features
            for feature, weight in self.risk_weights.items():
                if feature in engineered_features:
                    feature_risk = engineered_features[feature] * weight
                    risk_score += feature_risk
                    
                    if engineered_features[feature] > 0.7:
                        findings.append(f"High {feature}: {engineered_features[feature]:.2f}")
            
            # Additional pattern analysis
            amount = transaction.get('amount', 0)
            merchant_category = transaction.get('merchant_category', 'unknown')
            
            # High-risk merchant categories
            high_risk_categories = ['gambling', 'cryptocurrency', 'adult']
            if merchant_category in high_risk_categories:
                findings.append(f"High-risk merchant category: {merchant_category}")
                risk_score += 0.3
            
            # Round amount patterns
            if amount > 0 and amount % 100 == 0:
                findings.append(f"Round amount pattern: ${amount:.2f}")
                risk_score += 0.15
            
            risk_score = min(risk_score, 1.0)
            
            return AgentReport(
                agent_name=self.name,
                risk_score=risk_score,
                confidence=0.75,
                findings=findings,
                recommendations=["Escalate if risk > 0.6"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='SUCCESS'
            )
            
        except Exception as e:
            return AgentReport(
                agent_name=self.name,
                risk_score=0.0,
                confidence=0.0,
                findings=[f"Pattern analysis error: {str(e)}"],
                recommendations=["Manual pattern review required"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='ERROR'
            )

class BehaviorChecker:
    """Compares to user history for behavioral anomalies"""
    
    def __init__(self):
        self.name = "BehaviorChecker"
        self.behavioral_thresholds = {
            'amount_deviation': 2.0,
            'location_change': True,
            'merchant_deviation': 0.8
        }
    
    def analyze_transaction(self, transaction: Dict, user_history: pd.DataFrame) -> AgentReport:
        """Analyze transaction against user's historical behavior"""
        
        start_time = datetime.now()
        findings = []
        risk_score = 0.0
        
        try:
            if len(user_history) == 0:
                return AgentReport(
                    agent_name=self.name,
                    risk_score=0.0,
                    confidence=0.0,
                    findings=["No user history available"],
                    recommendations=["Cannot assess behavioral risk without history"],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    status='WARNING'
                )
            
            current_amount = transaction.get('amount', 0)
            current_location = transaction.get('location', 'unknown')
            current_merchant = transaction.get('merchant', 'unknown')
            
            # Analyze amount patterns
            historical_amounts = user_history['amount'].values
            amount_mean = np.mean(historical_amounts)
            amount_std = np.std(historical_amounts)
            
            if amount_std > 0:
                amount_zscore = abs(current_amount - amount_mean) / amount_std
                if amount_zscore > self.behavioral_thresholds['amount_deviation']:
                    findings.append(f"Unusual amount: {amount_zscore:.1f} standard deviations from mean")
                    risk_score += 0.4
            
            # Analyze location patterns
            historical_locations = user_history['location'].unique()
            if current_location not in historical_locations:
                findings.append(f"New location: {current_location}")
                risk_score += 0.3
            
            # Analyze merchant patterns
            historical_merchants = user_history['merchant'].unique()
            if current_merchant not in historical_merchants:
                findings.append(f"New merchant: {current_merchant}")
                risk_score += 0.2
            
            risk_score = min(risk_score, 1.0)
            
            return AgentReport(
                agent_name=self.name,
                risk_score=risk_score,
                confidence=0.85,
                findings=findings,
                recommendations=["Verify identity if risk > 0.5"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='SUCCESS'
            )
            
        except Exception as e:
            return AgentReport(
                agent_name=self.name,
                risk_score=0.0,
                confidence=0.0,
                findings=[f"Behavior analysis error: {str(e)}"],
                recommendations=["Manual behavioral review required"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='ERROR'
            )

class CaseResearcher:
    """Finds similar past fraud cases"""
    
    def __init__(self):
        self.name = "CaseResearcher"
        self.similarity_threshold = 0.7
    
    def analyze_transaction(self, transaction: Dict, user_history: pd.DataFrame) -> AgentReport:
        """Find similar fraud cases from history"""
        
        start_time = datetime.now()
        findings = []
        risk_score = 0.0
        
        try:
            if len(user_history) == 0:
                return AgentReport(
                    agent_name=self.name,
                    risk_score=0.0,
                    confidence=0.0,
                    findings=["No user history available for case research"],
                    recommendations=["Cannot find similar cases without history"],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    status='WARNING'
                )
            
            current_amount = transaction.get('amount', 0)
            current_merchant = transaction.get('merchant', 'unknown')
            
            # Find similar transactions in history
            similar_transactions = []
            
            for _, historical_tx in user_history.iterrows():
                similarity_score = 0
                
                # Amount similarity (within 20%)
                hist_amount = historical_tx.get('amount', 0)
                if abs(current_amount - hist_amount) / max(current_amount, hist_amount, 1) < 0.2:
                    similarity_score += 0.5
                
                # Merchant similarity
                hist_merchant = historical_tx.get('merchant', 'unknown')
                if current_merchant == hist_merchant:
                    similarity_score += 0.5
                
                if similarity_score >= self.similarity_threshold:
                    similar_transactions.append({
                        'transaction': historical_tx,
                        'similarity_score': similarity_score,
                        'was_fraud': historical_tx.get('is_fraud', False)
                    })
            
            # Analyze similar transactions
            if similar_transactions:
                fraud_cases = [tx for tx in similar_transactions if tx['was_fraud']]
                total_similar = len(similar_transactions)
                fraud_similar = len(fraud_cases)
                
                if fraud_similar > 0:
                    findings.append(f"Found {fraud_similar} similar fraud cases out of {total_similar} similar transactions")
                    fraud_ratio = fraud_similar / total_similar
                    risk_score += fraud_ratio * 0.6
            
            risk_score = min(risk_score, 1.0)
            
            return AgentReport(
                agent_name=self.name,
                risk_score=risk_score,
                confidence=0.75,
                findings=findings,
                recommendations=["Cross-reference with known fraud patterns"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='SUCCESS'
            )
            
        except Exception as e:
            return AgentReport(
                agent_name=self.name,
                risk_score=0.0,
                confidence=0.0,
                findings=[f"Case research error: {str(e)}"],
                recommendations=["Manual case research required"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                status='ERROR'
            )

class AlertGenerator:
    """Creates final risk report and alerts"""
    
    def __init__(self):
        self.name = "AlertGenerator"
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.85
        }
    
    def generate_final_report(self, agent_reports: List[AgentReport], 
                            transaction: Dict, user_profile: Dict = None) -> Dict[str, Any]:
        """Generate final comprehensive risk report"""
        
        start_time = datetime.now()
        
        try:
            # Combine agent reports
            total_risk_score = 0.0
            weighted_risk_score = 0.0
            total_confidence = 0.0
            all_findings = []
            all_recommendations = []
            
            # Calculate weighted risk score
            agent_weights = {
                'TransactionMonitor': 0.25,
                'PatternAnalyzer': 0.30,
                'BehaviorChecker': 0.25,
                'CaseResearcher': 0.20
            }
            
            for report in agent_reports:
                if report.status == 'SUCCESS':
                    total_risk_score += report.risk_score
                    weighted_risk_score += report.risk_score * agent_weights.get(report.agent_name, 0.25)
                    total_confidence += report.confidence
                    all_findings.extend(report.findings)
                    all_recommendations.extend(report.recommendations)
            
            # Calculate final metrics
            agent_count = len([r for r in agent_reports if r.status == 'SUCCESS'])
            avg_confidence = total_confidence / max(agent_count, 1)
            
            # Determine risk level
            if weighted_risk_score >= self.risk_thresholds['CRITICAL']:
                risk_level = 'CRITICAL'
                alert_priority = 'IMMEDIATE'
            elif weighted_risk_score >= self.risk_thresholds['HIGH']:
                risk_level = 'HIGH'
                alert_priority = 'HIGH'
            elif weighted_risk_score >= self.risk_thresholds['MEDIUM']:
                risk_level = 'MEDIUM'
                alert_priority = 'MEDIUM'
            elif weighted_risk_score >= self.risk_thresholds['LOW']:
                risk_level = 'LOW'
                alert_priority = 'LOW'
            else:
                risk_level = 'MINIMAL'
                alert_priority = 'LOW'
            
            # Generate action items
            action_items = []
            if risk_level in ['HIGH', 'CRITICAL']:
                action_items.extend([
                    "Immediate transaction review required",
                    "Contact user for verification",
                    "Consider temporary account hold"
                ])
            elif risk_level == 'MEDIUM':
                action_items.extend([
                    "Enhanced monitoring activated",
                    "Request additional authentication"
                ])
            else:
                action_items.extend([
                    "Continue standard monitoring",
                    "Log for pattern analysis"
                ])
            
            # Create final report
            final_report = {
                'final_risk_score': weighted_risk_score,
                'risk_level': risk_level,
                'alert_priority': alert_priority,
                'findings': all_findings[:10],
                'recommendations': list(set(all_recommendations))[:8],
                'action_items': action_items,
                'agent_reports': [
                    {
                        'agent_name': r.agent_name,
                        'risk_score': r.risk_score,
                        'confidence': r.confidence,
                        'status': r.status
                    } for r in agent_reports
                ],
                'processing_metrics': {
                    'total_processing_time': (datetime.now() - start_time).total_seconds(),
                    'agent_count': agent_count,
                    'avg_agent_confidence': avg_confidence
                },
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'status': 'COMPLETED'
            }
            
            return final_report
            
        except Exception as e:
            return {
                'final_risk_score': 0.0,
                'risk_level': 'ERROR',
                'alert_priority': 'LOW',
                'findings': [f"Alert generation error: {str(e)}"],
                'recommendations': ["Manual review required"],
                'action_items': ["Immediate manual intervention needed"],
                'agent_reports': [],
                'processing_metrics': {
                    'total_processing_time': (datetime.now() - start_time).total_seconds(),
                    'agent_count': 0,
                    'avg_agent_confidence': 0.0
                },
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'status': 'ERROR'
            }

# Simple orchestrator
class SimpleAIOrchestrator:
    """Simple orchestrator for the 5 AI agents"""
    
    def __init__(self):
        self.agents = {
            'TransactionMonitor': TransactionMonitor(),
            'PatternAnalyzer': PatternAnalyzer(),
            'BehaviorChecker': BehaviorChecker(),
            'CaseResearcher': CaseResearcher(),
            'AlertGenerator': AlertGenerator()
        }
    
    def analyze_transaction(self, transaction: Dict, user_history: pd.DataFrame = None, 
                          engineered_features: Dict = None) -> Dict[str, Any]:
        """Analyze transaction using all agents"""
        
        start_time = datetime.now()
        
        # Run all agents
        agent_reports = []
        
        # Transaction Monitor
        if user_history is not None:
            report = self.agents['TransactionMonitor'].analyze_transaction(transaction, user_history)
            agent_reports.append(report)
        
        # Pattern Analyzer
        if engineered_features is None:
            engineered_features = {
                'credit_risk_score': np.random.uniform(0, 0.5),
                'velocity_24h': np.random.uniform(0, 0.3),
                'velocity_4w': np.random.uniform(0, 0.2),
                'email_device_risk': np.random.uniform(0, 0.1),
                'age_velocity_interaction': np.random.uniform(0, 0.4)
            }
        
        report = self.agents['PatternAnalyzer'].analyze_transaction(transaction, engineered_features)
        agent_reports.append(report)
        
        # Behavior Checker
        if user_history is not None:
            report = self.agents['BehaviorChecker'].analyze_transaction(transaction, user_history)
            agent_reports.append(report)
        
        # Case Researcher
        if user_history is not None:
            report = self.agents['CaseResearcher'].analyze_transaction(transaction, user_history)
            agent_reports.append(report)
        
        # Generate final report
        final_report = self.agents['AlertGenerator'].generate_final_report(
            agent_reports, transaction
        )
        
        final_report['orchestration_metadata'] = {
            'total_processing_time': (datetime.now() - start_time).total_seconds(),
            'agents_used': len(agent_reports),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return final_report

# Demo function
def run_ai_demo():
    """Run a demonstration of the AI system"""
    
    print("🎯 AI Fraud Detection System Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    print("🤖 Initializing AI agents...")
    orchestrator = SimpleAIOrchestrator()
    
    # Create sample data
    print("\n📋 Creating sample transaction...")
    transaction = {
        'transaction_id': f'TX_{np.random.randint(100000, 999999)}',
        'user_id': f'USER_{np.random.randint(1000, 9999)}',
        'amount': np.random.uniform(100, 5000),
        'timestamp': datetime.now(),
        'merchant': f'MERCHANT_{np.random.randint(100, 999)}',
        'merchant_category': np.random.choice(['retail', 'online', 'service', 'gambling']),
        'country': np.random.choice(['US', 'CA', 'GB', 'CN', 'RU']),
        'device_info': 'Chrome/Windows',
        'location': f'CITY_{np.random.randint(1, 100)}'
    }
    
    # Create sample user history
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    user_history = pd.DataFrame({
        'transaction_id': [f'TX_{i}' for i in range(20)],
        'amount': np.random.lognormal(mean=4.0, sigma=1.0, size=20),
        'timestamp': dates,
        'merchant': np.random.choice(['MERCHANT_A', 'MERCHANT_B', 'MERCHANT_C'], 20),
        'location': np.random.choice(['CITY_1', 'CITY_2', 'CITY_3'], 20),
        'is_fraud': np.random.choice([False, True], 20, p=[0.95, 0.05])
    })
    
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
    
    print(f"\n📈 Agent Performance:")
    for agent_report in result['agent_reports']:
        print(f"  • {agent_report['agent_name']}: Risk={agent_report['risk_score']:.2f}, Status={agent_report['status']}")
    
    return result

if __name__ == "__main__":
    # Run demo
    result = run_ai_demo()
    print("\n✅ AI System Demo Completed!")
    print("All 5 AI agents are working together to detect fraud!")