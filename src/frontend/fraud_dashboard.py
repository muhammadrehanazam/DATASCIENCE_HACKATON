import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go # type: ignore
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
from collections import deque
from typing import Dict, List, Optional
import asyncio
import json
from websocket_dashboard_client import WebSocketDashboardClient

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.transactions = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.websocket_client = WebSocketDashboardClient()
        self.websocket_connected = False
        self.setup_websocket_callbacks()
        
    def setup_websocket_callbacks(self):
        """Setup WebSocket callback functions"""
        def on_alert(alert_data):
            self.alerts.append(alert_data)
            st.session_state.alerts = list(self.alerts)
            
        def on_transaction(transaction_data):
            self.transactions.append(transaction_data)
            st.session_state.transactions = list(self.transactions)
            
        self.websocket_client.add_alert_callback(on_alert)
        self.websocket_client.add_transaction_callback(on_transaction)
        
    def start_websocket(self):
        """Start WebSocket connection"""
        try:
            self.websocket_client.start()
            self.websocket_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to start WebSocket: {e}")
            return False
        
    def get_transaction_history(self, limit: int = 100) -> List[Dict]:
        try:
            response = requests.get(f"{self.api_base_url}/transactions/history?limit={limit}")
            if response.status_code == 200:
                return response.json().get("transactions", [])
            return []
        except Exception as e:
            st.error(f"Error fetching transaction history: {e}")
            return []
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent fraud alerts from API or return stored alerts"""
        try:
            response = requests.get(f"{self.api_base_url}/alerts/recent?limit={limit}")
            if response.status_code == 200:
                return response.json().get("alerts", [])
            return list(self.alerts)[:limit]  # Fallback to stored alerts
        except Exception as e:
            # Fallback to stored alerts if API fails
            return list(self.alerts)[:limit]
            
    def analyze_transaction(self, transaction_data: dict) -> dict:
        try:
            # Map dashboard fields to API expected fields
            api_data = {
                "amount": transaction_data.get("amount", 0),
                "merchant_id": transaction_data.get("merchant", "unknown"),
                "merchant_category": transaction_data.get("merchant_category", "retail"),
                "card_type": "credit",  # Default value
                "hour_of_day": 12,  # Default value
                "day_of_week": 0,   # Default value
                "month": 1,         # Default value
                "age": 30,          # Default value
                "gender": "M",      # Default value
                "city_population": "medium",  # Default value
                "credit_risk_score": 500,     # Default value
                "velocity_4w": 10,            # Default value
                "velocity_24h": 1,            # Default value
                "email_device_risk": 0,       # Default value
                "age_velocity_interaction": 30,  # Default value
                "country": transaction_data.get("country", "US"),
                "device_info": "web",
                "location": transaction_data.get("country", "US"),
                "payment_type": "card"
            }
            
            response = requests.post(
                f"{self.api_base_url}/analyze",
                json=api_data
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Analysis failed: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

def create_risk_gauge(risk_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ]
        }
    ))
    return fig

def main():
    st.title("🛡️ AI Fraud Detection Dashboard")
    
    dashboard = FraudDashboard()
    
    # Initialize session state
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
    if 'websocket_started' not in st.session_state:
        st.session_state.websocket_started = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # WebSocket connection status
        if not st.session_state.websocket_started:
            if st.button("🟢 Connect to Real-time Updates"):
                if dashboard.start_websocket():
                    st.session_state.websocket_started = True
                    st.success("Connected to real-time updates!")
                    st.rerun()
        else:
            st.success("🟢 Connected to real-time updates")
            if st.button("🔴 Disconnect"):
                dashboard.websocket_client.stop()
                st.session_state.websocket_started = False
                st.rerun()
        
        auto_refresh = st.toggle("Auto Refresh", value=True)
        search_term = st.text_input("Search Transactions", "")
        risk_filter = st.selectbox("Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])
        
        # Test transaction
        st.header("🧪 Test Transaction")
        with st.form("test_transaction"):
            amount = st.number_input("Amount ($)", min_value=0.01, value=100.00)
            merchant = st.text_input("Merchant", "Test Merchant")
            category = st.selectbox("Category", ["grocery", "restaurant", "retail", "online", "gas"])
            country = st.text_input("Country", "US")
            
            submitted = st.form_submit_button("Analyze")
            if submitted:
                test_data = {
                    "amount": amount,
                    "merchant": merchant,
                    "merchant_category": category,
                    "country": country
                }
                result = dashboard.analyze_transaction(test_data)
                st.session_state.selected_result = result
                st.session_state.test_data = test_data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Use real-time data if available, otherwise fetch
    if st.session_state.transactions:
        transactions = pd.DataFrame(st.session_state.transactions)
    else:
        transactions = pd.DataFrame(dashboard.get_transaction_history())
    
    if st.session_state.alerts:
        alerts = st.session_state.alerts
    else:
        alerts = dashboard.get_recent_alerts()
    
    # Ensure transactions is not None and handle empty DataFrames
    if transactions is None or transactions.empty:
        transactions = pd.DataFrame()
    
    with col1:
        total_transactions = len(transactions)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        fraud_count = len(transactions[transactions['fraud_probability'] > 0.7]) if len(transactions) > 0 else 0
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    
    with col3:
        high_risk_count = len(transactions[transactions['fraud_probability'] > 0.8]) if len(transactions) > 0 else 0
        st.metric("High Risk", f"{high_risk_count}")
    
    with col4:
        st.metric("Active Alerts", len(alerts))
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Feed", "📈 Risk Charts", "🔍 Analysis"])
    
    with tab1:
        st.header("📊 Live Transaction Feed")
        
        # Real-time alerts
        if st.session_state.alerts:
            st.subheader("🚨 Recent Alerts")
            for alert in st.session_state.alerts[-5:]:  # Show last 5 alerts
                alert_color = "🔴" if alert.get('severity') == 'high' else "🟡"
                st.write(f"{alert_color} {alert.get('message', 'Unknown alert')}")
        
        # Transaction table with real-time data
        if st.session_state.transactions:
            transactions_df = pd.DataFrame(st.session_state.transactions)
        else:
            transactions_df = pd.DataFrame(dashboard.get_transaction_history())
        
        # Ensure transactions_df is not None
        if transactions_df is None or transactions_df.empty:
            transactions_df = pd.DataFrame()
        
        if len(transactions_df) > 0:
            # Apply filters
            if search_term:
                mask = transactions_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                transactions_df = transactions_df[mask]
            
            if risk_filter != "All":
                risk_threshold = {"High Risk": 0.7, "Medium Risk": 0.4, "Low Risk": 0.1}
                if risk_filter in risk_threshold:
                    threshold = risk_threshold[risk_filter]
                    transactions_df = transactions_df[transactions_df['fraud_probability'] > threshold]
            
            st.dataframe(transactions_df, use_container_width=True)
        else:
            st.info("No transactions available. Connect to real-time updates or run a test transaction.")
    
    with tab2:
        st.subheader("📊 Risk Analysis")
        
        if not transactions.empty:
            risk_scores = [t.get('fraud_probability', 0) * 100 for t in transactions.to_dict('records')]
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_risk = np.mean(risk_scores)
                fig = create_risk_gauge(avg_risk / 100)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = px.histogram(
                    x=risk_scores,
                    nbins=20,
                    title="Risk Score Distribution"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Timeline
            df = pd.DataFrame(transactions)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Timestamp.now()))
                df['fraud_probability'] = df.get('fraud_probability', 0) * 100
                
                timeline_fig = px.scatter(
                    df,
                    x='timestamp',
                    y='fraud_probability',
                    size='amount' if 'amount' in df.columns else None,
                    color='fraud_probability',
                    color_continuous_scale='RdYlGn_r',
                    title='Transaction Risk Timeline'
                )
                st.plotly_chart(timeline_fig, width='stretch')
    
    with tab3:
        st.subheader("🔍 Transaction Analysis")
        
        if 'selected_result' in st.session_state:
            result = st.session_state.selected_result
            test_data = st.session_state.get('test_data', {})
            
            # Handle both API response format and expected format
            if 'error' in result:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            else:
                # Extract data from API response format
                fraud_probability = result.get('fraud_probability', 0)
                risk_level = result.get('risk_level', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                processing_time = result.get('processing_time', 0)
                recommendations = result.get('recommendations', [])
                
                # Map risk level to display format
                risk_display = risk_level.replace('_', ' ').title()
                risk_color = "🔴" if 'HIGH' in risk_level else "🟡" if 'MEDIUM' in risk_level else "🟢"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Transaction Details**")
                    st.write(f"Amount: ${test_data.get('amount', 0):.2f}")
                    st.write(f"Merchant: {test_data.get('merchant', 'N/A')}")
                    st.write(f"Category: {test_data.get('merchant_category', 'N/A')}")
                    st.write(f"Country: {test_data.get('country', 'N/A')}")
                
                with col2:
                    st.markdown("**Analysis Results**")
                    st.write(f"Risk Level: {risk_color} {risk_display}")
                    st.write(f"Fraud Probability: {fraud_probability:.1%}")
                    st.write(f"Confidence: {confidence:.1%}")
                    st.write(f"Processing Time: {processing_time:.2f}s")
                
                # ML vs AI Breakdown (if available)
                if 'ml_probability' in result or 'ai_risk_score' in result:
                    st.markdown("**ML vs AI Analysis**")
                    ml_col1, ml_col2 = st.columns(2)
                    
                    with ml_col1:
                        if 'ml_probability' in result:
                            st.write(f"ML Probability: {result.get('ml_probability', 0):.3f}")
                    with ml_col2:
                        if 'ai_risk_score' in result:
                            st.write(f"AI Risk Score: {result.get('ai_risk_score', 0):.3f}")
                
                # Recommendations
                if recommendations:
                    st.markdown("**Recommendations**")
                    for i, rec in enumerate(recommendations[:3], 1):
                        st.write(f"{i}. {rec}")
                
                # Transaction ID (if available)
                if 'transaction_id' in result:
                    st.markdown("**Transaction ID**")
                    st.code(result.get('transaction_id'))
        else:
            st.info("Analyze a test transaction to see detailed results.")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()