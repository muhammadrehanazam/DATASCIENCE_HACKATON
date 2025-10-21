"""
WebSocket Server for Real-time Fraud Alerts
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketAlertServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register_client(self, websocket, path=None):
        """Register a new WebSocket client and handle messages"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'test_alert':
                        logger.info("Received test alert request")
                        await self.send_test_alert()
                        # Send confirmation back
                        await websocket.send(json.dumps({
                            'type': 'test_confirmation',
                            'message': 'Test alert sent successfully',
                            'timestamp': datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_alert(self, alert_data):
        """Broadcast fraud alert to all connected clients"""
        if not self.clients:
            logger.info("No clients connected to broadcast alert")
            return
        
        message = json.dumps({
            'type': 'fraud_alert',
            'timestamp': datetime.now().isoformat(),
            'data': alert_data
        })
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send(message)
                logger.info(f"Alert sent to client: {alert_data.get('title', 'Fraud Alert')}")
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error sending alert to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
    
    async def send_test_alert(self):
        """Send a test alert to all connected clients"""
        test_alert = {
            'alert_id': f"TEST_{uuid.uuid4().hex[:8].upper()}",
            'transaction_id': f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'alert_type': 'TEST_ALERT',
            'severity': 'HIGH',
            'title': 'Test Fraud Alert',
            'description': 'This is a test alert from the fraud detection system',
            'fraud_probability': 0.85,
            'risk_level': 'High'
        }
        
        await self.broadcast_alert(test_alert)
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.register_client, self.host, self.port):
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

# Global server instance
alert_server = WebSocketAlertServer()

async def send_fraud_alert(transaction_data: dict, fraud_probability: float, risk_level: str):
    """Send a fraud alert to all connected WebSocket clients"""
    alert_data = {
        'alert_id': f"ALERT_{uuid.uuid4().hex[:8].upper()}",
        'transaction_id': transaction_data.get('transaction_id', 'UNKNOWN'),
        'alert_type': 'HIGH_RISK_TRANSACTION',
        'severity': 'HIGH' if fraud_probability > 0.5 else 'MEDIUM',
        'title': f"High Risk Transaction Detected - {risk_level} Risk",
        'description': f"Transaction flagged with {fraud_probability:.1%} fraud probability. Amount: ${transaction_data.get('amount', 0):.2f}",
        'fraud_probability': fraud_probability,
        'risk_level': risk_level,
        'transaction_amount': transaction_data.get('amount', 0),
        'merchant_category': transaction_data.get('merchant_category', 'Unknown')
    }
    
    await alert_server.broadcast_alert(alert_data)

async def main():
    """Main function to run the WebSocket server"""
    await alert_server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("WebSocket server stopped")