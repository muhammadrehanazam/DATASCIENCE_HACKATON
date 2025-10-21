"""
Database models for fraud detection system
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    merchant_id = Column(String(100), nullable=False)
    merchant_category = Column(String(50), nullable=False)
    card_type = Column(String(20), nullable=False)
    hour_of_day = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    city_population = Column(String(20), nullable=False)
    credit_risk_score = Column(Integer, nullable=False)
    velocity_4w = Column(Integer, nullable=False)
    velocity_24h = Column(Integer, nullable=False)
    email_device_risk = Column(Integer, nullable=False)
    age_velocity_interaction = Column(Integer, nullable=False)
    country = Column(String(10), nullable=True)
    device_info = Column(String(100), nullable=True)
    location = Column(String(50), nullable=True)
    payment_type = Column(String(20), nullable=True)
    
    # Analysis results
    fraud_probability = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    confidence = Column(Float, nullable=True)
    is_fraud = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime, nullable=True)

class AgentDecision(Base):
    __tablename__ = "agent_decisions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    findings = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    labels = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

# Database connection and session management
class DatabaseManager:
    def __init__(self, database_url="sqlite:///fraud_detection.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        session.close()
    
    def save_transaction(self, transaction_data: dict, analysis_result: dict = None) -> str:
        """Save transaction and analysis results to database"""
        session = self.get_session()
        try:
            # Create transaction record
            transaction = Transaction(
                transaction_id=transaction_data.get('transaction_id'),
                amount=transaction_data.get('amount', 0.0),
                merchant_id=transaction_data.get('merchant_id', ''),
                merchant_category=transaction_data.get('merchant_category', ''),
                card_type=transaction_data.get('card_type', ''),
                hour_of_day=transaction_data.get('hour_of_day', 0),
                day_of_week=transaction_data.get('day_of_week', 0),
                month=transaction_data.get('month', 0),
                age=transaction_data.get('age', 0),
                gender=transaction_data.get('gender', ''),
                city_population=transaction_data.get('city_population', ''),
                credit_risk_score=transaction_data.get('credit_risk_score', 0),
                velocity_4w=transaction_data.get('velocity_4w', 0),
                velocity_24h=transaction_data.get('velocity_24h', 0),
                email_device_risk=transaction_data.get('email_device_risk', 0),
                age_velocity_interaction=transaction_data.get('age_velocity_interaction', 0),
                country=transaction_data.get('country'),
                device_info=transaction_data.get('device_info'),
                location=transaction_data.get('location'),
                payment_type=transaction_data.get('payment_type')
            )
            
            # Add analysis results if available
            if analysis_result:
                if 'integrated_analysis' in analysis_result:
                    integrated = analysis_result['integrated_analysis']
                    transaction.fraud_probability = integrated.get('fraud_probability')
                    transaction.risk_level = integrated.get('risk_level')
                    transaction.confidence = integrated.get('confidence')
                    transaction.analyzed_at = datetime.utcnow()
            
            session.add(transaction)
            session.commit()
            
            return transaction.transaction_id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def save_agent_decision(self, transaction_id: str, agent_result: dict) -> int:
        """Save agent decision to database"""
        session = self.get_session()
        try:
            agent_decision = AgentDecision(
                transaction_id=transaction_id,
                agent_name=agent_result.get('agent_name', ''),
                risk_score=agent_result.get('risk_score', 0.0),
                risk_level=agent_result.get('risk_level', ''),
                status=agent_result.get('status', ''),
                confidence=agent_result.get('confidence'),
                findings=agent_result.get('findings'),
                recommendations=agent_result.get('recommendations'),
                processing_time=agent_result.get('processing_time')
            )
            
            session.add(agent_decision)
            session.commit()
            
            return agent_decision.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def save_alert(self, transaction_id: str, alert_data: dict) -> str:
        """Save fraud alert to database"""
        session = self.get_session()
        try:
            import uuid
            alert_id = f"ALERT_{uuid.uuid4().hex[:8].upper()}"
            
            alert = Alert(
                alert_id=alert_id,
                transaction_id=transaction_id,
                alert_type=alert_data.get('alert_type', 'FRAUD_DETECTION'),
                severity=alert_data.get('severity', 'MEDIUM'),
                title=alert_data.get('title', 'Fraud Alert'),
                description=alert_data.get('description', ''),
                fraud_probability=alert_data.get('fraud_probability', 0.0),
                risk_level=alert_data.get('risk_level', ''),
                is_active=True
            )
            
            session.add(alert)
            session.commit()
            
            return alert.alert_id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def save_model_prediction(self, transaction_id: str, model_name: str, prediction: dict) -> int:
        """Save model prediction to database"""
        session = self.get_session()
        try:
            model_pred = ModelPrediction(
                transaction_id=transaction_id,
                model_name=model_name,
                fraud_probability=prediction.get('fraud_probability', 0.0),
                confidence=prediction.get('confidence'),
                processing_time=prediction.get('processing_time')
            )
            
            session.add(model_pred)
            session.commit()
            
            return model_pred.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_active_alerts(self, limit: int = 100) -> list:
        """Get active fraud alerts"""
        session = self.get_session()
        try:
            alerts = session.query(Alert).filter(Alert.is_active == True).order_by(Alert.created_at.desc()).limit(limit).all()
            return [{
                'alert_id': alert.alert_id,
                'transaction_id': alert.transaction_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'fraud_probability': alert.fraud_probability,
                'risk_level': alert.risk_level,
                'created_at': alert.created_at.isoformat()
            } for alert in alerts]
        finally:
            self.close_session(session)
    
    def get_transaction_history(self, transaction_id: str = None, limit: int = 50) -> list:
        """Get transaction history"""
        session = self.get_session()
        try:
            query = session.query(Transaction)
            
            if transaction_id:
                query = query.filter(Transaction.transaction_id == transaction_id)
            
            transactions = query.order_by(Transaction.created_at.desc()).limit(limit).all()
            
            return [{
                'transaction_id': tx.transaction_id,
                'amount': tx.amount,
                'merchant_id': tx.merchant_id,
                'merchant_category': tx.merchant_category,
                'fraud_probability': tx.fraud_probability,
                'risk_level': tx.risk_level,
                'confidence': tx.confidence,
                'is_fraud': tx.is_fraud,
                'created_at': tx.created_at.isoformat(),
                'analyzed_at': tx.analyzed_at.isoformat() if tx.analyzed_at else None
            } for tx in transactions]
        finally:
            self.close_session(session)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a fraud alert"""
        session = self.get_session()
        try:
            alert = session.query(Alert).filter(Alert.alert_id == alert_id).first()
            if alert:
                alert.is_active = False
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)

# Global database manager instance
db_manager = DatabaseManager()