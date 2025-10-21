import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import requests
import json
from integrated_fraud_system import IntegratedFraudDetector

# Set page config
st.set_page_config(
    page_title="AI-Powered Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Load data
def load_data():
    """Load fraud detection data"""
    try:
        # Load ensemble results - the first column contains model names
        ensemble_results = pd.read_csv('ensemble_comparison_results.csv', index_col=0)
        ensemble_results.index.name = 'Model'
        ensemble_results = ensemble_results.reset_index()  # Convert index to column
        
        # Load explanation data
        with open('model_explanation_data.pkl', 'rb') as f:
            explanation_data = pickle.load(f)
        
        return ensemble_results, explanation_data
    except:
        # Create sample data
        ensemble_results = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble'],
            'accuracy': [0.79, 0.99, 0.96, 0.97],
            'precision': [0.05, 0.00, 0.11, 0.17],
            'recall': [0.81, 0.00, 0.32, 0.28],
            'f1_score': [0.09, 0.00, 0.17, 0.21],
            'auc': [0.87, 0.77, 0.83, 0.87]
        })
        
        explanation_data = {
            'feature_importance': pd.DataFrame({
                'feature': ['credit_risk_score', 'velocity_4w', 'email_device_risk', 'age_velocity_interaction', 'velocity_24h'],
                'importance': [0.25, 0.20, 0.15, 0.12, 0.10]
            })
        }
        
        return ensemble_results, explanation_data

# Test API endpoints
def test_api_endpoints():
    """Test both traditional ML and integrated endpoints"""
    test_transaction = {
        "amount": 1500.0,
        "merchant_category": "online",
        "country": "US",
        "payment_type": "credit",
        "customer_age": 35
    }
    
    results = {}
    
    # Test traditional ML endpoint
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_transaction,
            params={"integrated": "false", "ai_analysis": "false"}
        )
        results['traditional_ml'] = response.json() if response.status_code == 200 else None
    except Exception as e:
        results['traditional_ml'] = {"error": str(e)}
    
    # Test integrated endpoint
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_transaction,
            params={"integrated": "true", "ai_analysis": "true"}
        )
        results['integrated'] = response.json() if response.status_code == 200 else None
    except Exception as e:
        results['integrated'] = {"error": str(e)}
    
    return results

# Simulate real-time transaction analysis
def simulate_transaction_analysis():
    """Simulate analyzing a transaction with both ML and AI"""
    detector = IntegratedFraudDetector()
    
    # Create sample transaction
    transaction = {
        'transaction_id': f'TX_{np.random.randint(100000, 999999)}',
        'user_id': f'USER_{np.random.randint(1000, 9999)}',
        'amount': np.random.uniform(100, 10000),
        'timestamp': datetime.now(),
        'merchant': f'MERCHANT_{np.random.randint(100, 999)}',
        'merchant_category': np.random.choice(['retail', 'online', 'service', 'gambling', 'cryptocurrency']),
        'country': np.random.choice(['US', 'CA', 'GB', 'CN', 'RU']),
        'device_info': 'Chrome/Windows',
        'location': f'CITY_{np.random.randint(1, 100)}',
        'payment_type': np.random.choice(['credit', 'debit', 'transfer']),
        'customer_age': np.random.randint(18, 80)
    }
    
    # Create sample user history
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    user_history = pd.DataFrame({
        'transaction_id': [f'TX_{i}' for i in range(30)],
        'amount': np.random.lognormal(mean=4.0, sigma=1.0, size=30),
        'timestamp': dates,
        'merchant': np.random.choice(['MERCHANT_A', 'MERCHANT_B', 'MERCHANT_C'], 30),
        'location': np.random.choice(['CITY_1', 'CITY_2', 'CITY_3'], 30),
        'is_fraud': np.random.choice([False, True], 30, p=[0.95, 0.05])
    })
    
    # Run integrated analysis
    result = detector.analyze_transaction(transaction, user_history)
    
    return result, transaction

# Main dashboard
def main():
    st.title("🛡️ AI-Powered Fraud Detection Dashboard")
    st.markdown("### Combining ML Models with AI Agent Analysis")
    st.markdown("---")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Load data
    ensemble_results, explanation_data = load_data()
    
    # Test API endpoints
    with st.sidebar:
        if st.button("🧪 Test API Endpoints"):
            with st.spinner("Testing endpoints..."):
                api_results = test_api_endpoints()
                st.session_state['api_results'] = api_results
        
        # Simulate transaction analysis
        if st.button("🔍 Analyze Sample Transaction"):
            with st.spinner("Running integrated analysis..."):
                analysis_result, transaction_data = simulate_transaction_analysis()
                st.session_state['analysis_result'] = analysis_result
                st.session_state['transaction_data'] = transaction_data
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transactions", "1,000,000")
    
    with col2:
        st.metric("Fraud Detected", "11,000")
    
    with col3:
        st.metric("Fraud Rate", "1.1%")
    
    with col4:
        best_auc = ensemble_results['auc'].max()
        st.metric("Best AUC Score", f"{best_auc:.1%}")
    
    with col5:
        st.metric("AI Agents Active", "5")
    
    st.markdown("---")
    
    # API Test Results
    if 'api_results' in st.session_state:
        st.subheader("🧪 API Endpoint Test Results")
        api_results = st.session_state['api_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Traditional ML Endpoint**")
            if api_results['traditional_ml'] and 'error' not in api_results['traditional_ml']:
                pred = api_results['traditional_ml']['prediction']
                st.success(f"✅ Status: {pred['risk_level']} (Confidence: {pred['confidence']:.1%})")
                st.write(f"Fraud Probability: {pred['fraud_probability']:.1%}")
            else:
                st.error("❌ Traditional ML endpoint failed")
        
        with col2:
            st.markdown("**Integrated ML + AI Endpoint**")
            if api_results['integrated'] and 'error' not in api_results['integrated']:
                pred = api_results['integrated']['prediction']
                st.success(f"✅ Status: {pred['risk_level']} (Confidence: {pred['confidence']:.1%})")
                st.write(f"Fraud Probability: {pred['fraud_probability']:.1%}")
            else:
                st.error("❌ Integrated endpoint failed")
    
    # Integrated System Demo
    st.markdown("---")
    st.subheader("🤖 Integrated ML + AI Fraud Detection Demo")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Test Scenarios:**")
        
        # Predefined test scenarios
        test_scenarios = {
            "Normal Transaction": {
                "transaction_id": "DEMO_NORMAL_001",
                "amount": 50.0,
                "merchant_id": "SAFE_MERCHANT",
                "merchant_category": "grocery",
                "card_type": "visa",
                "hour_of_day": 14,
                "day_of_week": 3,
                "month": 6,
                "age": 35,
                "gender": "male",
                "city_population": "medium",
                "credit_risk_score": 650,
                "velocity_4w": 10,
                "velocity_24h": 2,
                "email_device_risk": 5,
                "age_velocity_interaction": 15
            },
            "Suspicious Transaction": {
                "transaction_id": "DEMO_FRAUD_001", 
                "amount": 2000.0,
                "merchant_id": "SUSPICIOUS_MERCHANT",
                "merchant_category": "online_gaming",
                "card_type": "visa",
                "hour_of_day": 3,
                "day_of_week": 1,
                "month": 6,
                "age": 22,
                "gender": "male",
                "city_population": "large",
                "credit_risk_score": 350,
                "velocity_4w": 150,
                "velocity_24h": 25,
                "email_device_risk": 80,
                "age_velocity_interaction": 120
            },
            "Edge Case Transaction": {
                "transaction_id": "DEMO_EDGE_001",
                "amount": 0.01,
                "merchant_id": "UNKNOWN_MERCHANT",
                "merchant_category": "cryptocurrency",
                "card_type": "mastercard",
                "hour_of_day": 0,
                "day_of_week": 1,
                "month": 2,
                "age": 99,
                "gender": "other",
                "city_population": "small",
                "credit_risk_score": 850,
                "velocity_4w": 0,
                "velocity_24h": 0,
                "email_device_risk": 0,
                "age_velocity_interaction": 0
            }
        }
        
        selected_scenario = st.selectbox("Choose a test scenario:", list(test_scenarios.keys()))
        
        if st.button("🚀 Run Integrated Analysis"):
            with st.spinner("Running ML models and AI agents..."):
                try:
                    # Import and run the integrated system
                    from integrated_fraud_system import IntegratedFraudDetector
                    detector = IntegratedFraudDetector()
                    
                    test_data = test_scenarios[selected_scenario]
                    result = detector.analyze_transaction(test_data)
                    
                    st.session_state['integrated_result'] = result
                    st.session_state['test_data'] = test_data
                    st.session_state['scenario_name'] = selected_scenario
                    
                except Exception as e:
                    st.error(f"Error running integrated analysis: {str(e)}")
    
    with col2:
        if 'integrated_result' in st.session_state:
            result = st.session_state['integrated_result']
            test_data = st.session_state['test_data']
            scenario_name = st.session_state['scenario_name']
            
            st.markdown(f"**Analysis Results for: {scenario_name}**")
            
            if result['status'] == 'SUCCESS':
                integrated_analysis = result['integrated_analysis']
                
                # Key metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    risk_color = "🔴" if integrated_analysis['risk_level'] in ['HIGH', 'CRITICAL'] else "🟡" if integrated_analysis['risk_level'] == 'MEDIUM' else "🟢"
                    st.metric("Risk Level", f"{risk_color} {integrated_analysis['risk_level']}")
                with col_b:
                    st.metric("Fraud Probability", f"{integrated_analysis['fraud_probability']:.1%}")
                with col_c:
                    st.metric("Confidence", f"{integrated_analysis['confidence']:.1%}")
                
                # ML vs AI breakdown
                with st.expander("🔍 ML vs AI Analysis Breakdown"):
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        st.write(f"**ML Models Contribution:** {integrated_analysis['ml_contribution']:.3f}")
                        if 'ml_predictions' in result:
                            ml_preds = result['ml_predictions']
                            final_pred = ml_preds.get('final_prediction', {})
                            if final_pred:
                                st.write(f"Final ML Prediction: {final_pred.get('fraud_probability', 0):.3f}")
                    
                    with col_y:
                        st.write(f"**AI Agents Contribution:** {integrated_analysis['ai_contribution']:.3f}")
                        if 'ai_analysis' in result:
                            ai_analysis = result['ai_analysis']
                            st.write(f"AI Risk Score: {ai_analysis.get('final_risk_score', 0):.3f}")
                            st.write(f"AI Risk Level: {ai_analysis.get('risk_level', 'N/A')}")
                
                # AI Agent details
                if 'ai_analysis' in result and 'agent_reports' in result['ai_analysis']:
                    with st.expander("🧠 AI Agent Analysis Details"):
                        agent_reports = result['ai_analysis']['agent_reports']
                        
                        for agent in agent_reports:
                            agent_color = "🔴" if agent['risk_score'] > 0.5 else "🟡" if agent['risk_score'] > 0.2 else "🟢"
                            st.write(f"**{agent_color} {agent['agent_name']}**")
                            st.write(f"Risk Score: {agent['risk_score']:.3f}")
                            st.write(f"Status: {agent['status']}")
                            
                            if 'findings' in agent and agent['findings']:
                                st.write("Findings:")
                                for finding in agent['findings']:
                                    st.write(f"• {finding}")
                            
                            if 'recommendations' in agent and agent['recommendations']:
                                st.write("Recommendations:")
                                for rec in agent['recommendations']:
                                    st.write(f"• {rec}")
                            st.write("---")
                
                # Recommendations
                if 'recommendations' in result and result['recommendations']:
                    with st.expander("💡 System Recommendations"):
                        for i, rec in enumerate(result['recommendations'], 1):
                            st.write(f"{i}. {rec}")
            
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

    # Transaction Analysis Results
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.subheader("🔍 Integrated Transaction Analysis")
        
        result = st.session_state['analysis_result']
        transaction = st.session_state['transaction_data']
        
        if result['status'] == 'SUCCESS':
            analysis = result['integrated_analysis']
            
            # Transaction details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Transaction Details**")
                st.write(f"Amount: ${transaction['amount']:.2f}")
                st.write(f"Merchant: {transaction['merchant']}")
                st.write(f"Category: {transaction['merchant_category']}")
                st.write(f"Country: {transaction['country']}")
            
            with col2:
                st.markdown("**🎯 Integrated Analysis Results**")
                risk_color = "🔴" if analysis['risk_level'] == 'HIGH' else "🟡" if analysis['risk_level'] == 'MEDIUM' else "🟢"
                st.write(f"Risk Level: {risk_color} {analysis['risk_level']}")
                st.write(f"Fraud Probability: {analysis['fraud_probability']:.1%}")
                st.write(f"Confidence: {analysis['confidence']:.1%}")
                st.write(f"Processing Time: {result['processing_time']:.2f}s")
            
            # ML vs AI Breakdown
            st.markdown("**🔍 ML vs AI Analysis Breakdown**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"ML Contribution: {analysis['ml_contribution']:.3f}")
                if 'ml_predictions' in result:
                    ml_preds = result['ml_predictions']
                    for model_name, pred in ml_preds.items():
                        if isinstance(pred, dict) and 'fraud_probability' in pred:
                            st.write(f"  {model_name}: {pred['fraud_probability']:.3f}")
            
            with col2:
                st.write(f"AI Contribution: {analysis['ai_contribution']:.3f}")
                if 'ai_analysis' in result and 'agent_reports' in result['ai_analysis']:
                    agent_reports = result['ai_analysis']['agent_reports']
                    for agent in agent_reports:
                        st.write(f"  {agent['agent_name']}: Risk={agent['risk_score']:.3f}")
            
            # AI Agent Analysis
            st.markdown("**🧠 AI Agent Analysis**")
            if 'ai_analysis' in result and 'agent_reports' in result['ai_analysis']:
                agent_reports = result['ai_analysis']['agent_reports']
                
                for agent in agent_reports:
                    with st.expander(f"{agent['agent_name']} - Risk: {agent['risk_score']:.3f}"):
                        st.write(f"**Status:** {agent['status']}")
                        st.write(f"**Confidence:** {agent.get('confidence', 'N/A')}")
                        if 'findings' in agent:
                            st.write("**Findings:**")
                            for finding in agent['findings']:
                                st.write(f"• {finding}")
                        if 'recommendations' in agent:
                            st.write("**Recommendations:**")
                            for rec in agent['recommendations']:
                                st.write(f"• {rec}")
            
            # Recommendations
            st.markdown("**💡 Recommendations**")
            for i, rec in enumerate(result['recommendations'][:5], 1):
                st.write(f"{i}. {rec}")
            
            # Visual breakdown
            st.markdown("**📊 Analysis Breakdown Visualization**")
            
            # Create pie chart for ML vs AI contribution
            labels = ['ML Models', 'AI Agents']
            values = [analysis['ml_contribution'], analysis['ai_contribution']]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title="ML vs AI Contribution to Risk Assessment")
            st.plotly_chart(fig, width="stretch")
    
    # Model performance section
    st.markdown("---")
    st.subheader("📊 Model Performance & Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance Comparison")
        
        fig = px.bar(
            ensemble_results,
            x='Model',
            y='auc',
            title="AUC Scores by Model",
            color='auc',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("🔍 Feature Importance")
        
        if 'feature_importance' in explanation_data:
            feature_imp = explanation_data['feature_importance'].head(10)
            
            fig = px.bar(
                feature_imp,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
    
    # Risk distribution
    st.subheader("📈 Risk Analysis")
    
    # Simulate risk scores
    np.random.seed(42)
    risk_scores = np.random.beta(2, 50, 1000)  # Skewed towards low risk
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=risk_scores,
            nbins=50,
            title="Fraud Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk categories
        risk_categories = pd.cut(
            risk_scores,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        ).value_counts()
        
        fig = px.pie(
            values=risk_categories.values,
            names=risk_categories.index,
            title="Risk Category Distribution",
            color_discrete_map={
                'Low Risk': '#4caf50',
                'Medium Risk': '#ff9800',
                'High Risk': '#f44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time alerts
    st.subheader("🚨 Recent Fraud Alerts")
    
    # Simulate recent alerts
    alerts = []
    current_time = datetime.now()
    
    for i in range(5):
        alert_time = current_time - timedelta(minutes=np.random.randint(1, 60))
        risk_score = np.random.uniform(0.8, 0.99)
        alerts.append({
            'Time': alert_time.strftime('%H:%M:%S'),
            'Risk Score': f"{risk_score:.1%}",
            'Status': '🔴 High Risk',
            'Analysis Type': np.random.choice(['ML Only', 'ML + AI'])
        })
    
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, width="stretch")
    
    # Model comparison table
    st.subheader("📋 Model Performance Summary")
    
    # Format the results table
    display_results = ensemble_results.round(3)
    st.dataframe(display_results, width="stretch")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "AI-Powered Fraud Detection Dashboard v2.0 | Combining ML Models with AI Agent Intelligence"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()