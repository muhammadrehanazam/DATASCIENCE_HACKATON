"""
Fraud Detection REST API Server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
import logging

# Import our existing modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.integrated_fraud_system import IntegratedFraudDetector
from database_models import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="REST API for fraud detection with ML and AI agents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
fraud_system = IntegratedFraudDetector()

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., min_length=1, description="Merchant ID")
    merchant_category: str = Field(..., min_length=1, description="Merchant category")
    card_type: str = Field(..., description="Card type (credit/debit)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    gender: str = Field(..., description="Gender (M/F)")
    city_population: str = Field(..., description="City population category")
    credit_risk_score: int = Field(..., ge=0, le=1000, description="Credit risk score")
    velocity_4w: int = Field(..., ge=0, description="4-week transaction velocity")
    velocity_24h: int = Field(..., ge=0, description="24-hour transaction velocity")
    email_device_risk: int = Field(..., ge=0, description="Email device risk score")
    age_velocity_interaction: int = Field(..., description="Age-velocity interaction")
    country: Optional[str] = Field(None, description="Country code")
    device_info: Optional[str] = Field(None, description="Device information")
    location: Optional[str] = Field(None, description="Location")
    payment_type: Optional[str] = Field(None, description="Payment type")

class TransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    confidence: float
    ml_probability: float
    ai_risk_score: float
    recommendations: List[str]
    processing_time: float
    status: str

class AlertResponse(BaseModel):
    alert_id: str
    transaction_id: str
    alert_type: str
    severity: str
    title: str
    description: Optional[str]
    fraud_probability: float
    risk_level: str
    created_at: datetime
    is_active: bool

# Helper functions
def generate_transaction_id() -> str:
    return f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8].upper()}"

def process_transaction(transaction_data: dict) -> Dict[str, Any]:
    """Process transaction and save results to database"""
    start_time = datetime.now()
    
    # Generate transaction ID
    transaction_id = generate_transaction_id()
    transaction_data['transaction_id'] = transaction_id
    
    try:
        # Run fraud detection analysis
        analysis_result = fraud_system.analyze_transaction(transaction_data)
        
        # Save transaction to database
        db_manager.save_transaction(transaction_data, analysis_result)
        
        # Check if we should generate alert
        fraud_probability = analysis_result['integrated_analysis']['fraud_probability']
        risk_level = analysis_result['integrated_analysis']['risk_level']
        
        if fraud_probability > 0.3 or risk_level in ['High', 'Very High']:
            alert_data = {
                'alert_type': 'HIGH_RISK_TRANSACTION',
                'severity': 'HIGH' if fraud_probability > 0.5 else 'MEDIUM',
                'title': f"High Risk Transaction Detected - {risk_level} Risk",
                'description': f"Transaction {transaction_id} flagged with {fraud_probability:.1%} fraud probability",
                'fraud_probability': fraud_probability,
                'risk_level': risk_level
            }
            
            db_manager.save_alert(transaction_id, alert_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'transaction_id': transaction_id,
            'analysis_result': analysis_result,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze",
            "/alerts",
            "/alerts/{alert_id}/acknowledge",
            "/transactions",
            "/transactions/{transaction_id}",
            "/health"
        ]
    }

@app.post("/analyze", response_model=TransactionResponse)
async def analyze_transaction(transaction: TransactionRequest):
    """Analyze a single transaction for fraud detection"""
    transaction_data = transaction.dict()
    
    result = process_transaction(transaction_data)
    analysis_result = result['analysis_result']
    
    return TransactionResponse(
        transaction_id=result['transaction_id'],
        fraud_probability=analysis_result['integrated_analysis']['fraud_probability'],
        risk_level=analysis_result['integrated_analysis']['risk_level'],
        confidence=analysis_result['integrated_analysis']['confidence'],
        ml_probability=analysis_result['ml_predictions'].get('final_prediction', {}).get('fraud_probability', 0.0),
        ai_risk_score=analysis_result['ai_analysis'].get('final_risk_score', 0.0),
        recommendations=analysis_result['recommendations'],
        processing_time=result['processing_time'],
        status="completed"
    )

@app.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(limit: int = 100):
    """Get active fraud alerts"""
    alerts = db_manager.get_active_alerts(limit)
    return [AlertResponse(**alert) for alert in alerts]

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str):
    """Acknowledge a fraud alert"""
    success = db_manager.acknowledge_alert(alert_id, acknowledged_by)
    if success:
        return {"message": "Alert acknowledged successfully"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/transactions")
async def get_transactions(limit: int = 50, transaction_id: Optional[str] = None):
    """Get transaction history"""
    transactions = db_manager.get_transaction_history(transaction_id, limit)
    return {"transactions": transactions, "count": len(transactions)}

@app.get("/transactions/history")
async def get_transaction_history(limit: int = 100):
    """Get transaction history (alias for /transactions)"""
    transactions = db_manager.get_transaction_history(limit=limit)
    return {"transactions": transactions, "count": len(transactions)}

@app.get("/alerts/recent")
async def get_recent_alerts(limit: int = 10):
    """Get recent fraud alerts"""
    alerts = db_manager.get_active_alerts(limit)
    return {"alerts": alerts, "count": len(alerts)}

@app.get("/transactions/{transaction_id}")
async def get_transaction_details(transaction_id: str):
    """Get detailed information about a specific transaction"""
    transactions = db_manager.get_transaction_history(transaction_id, limit=1)
    if not transactions:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return transactions[0]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "fraud_detection": "operational",
            "database": "operational"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    print("Starting Fraud Detection API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)