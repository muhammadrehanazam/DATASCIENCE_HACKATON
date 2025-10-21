import asyncio
import websockets
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable
import threading
from queue import Queue, Empty

class WebSocketDashboardClient:
    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.connected = False
        self.message_queue = Queue()
        self.alert_callbacks = []
        self.transaction_callbacks = []
        self.websocket = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def add_alert_callback(self, callback: Callable):
        """Add a callback function for fraud alerts"""
        self.alert_callbacks.append(callback)
        
    def add_transaction_callback(self, callback: Callable):
        """Add a callback function for transaction updates"""
        self.transaction_callbacks.append(callback)
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            self.reconnect_attempts = 0
            print(f"🟢 Connected to WebSocket server at {self.websocket_url}")
            
            # Register as dashboard client
            await self.websocket.send(json.dumps({
                "type": "register_dashboard",
                "client_type": "dashboard"
            }))
            
            # Start listening for messages
            await self.listen_for_messages()
            
        except Exception as e:
            print(f"🔴 Failed to connect to WebSocket: {e}")
            self.connected = False
            await self.handle_reconnection()
            
    async def handle_reconnection(self):
        """Handle reconnection attempts"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(2 ** self.reconnect_attempts, 30)  # Exponential backoff
            print(f"🟡 Reconnection attempt {self.reconnect_attempts} in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            await self.connect()
        else:
            print("🔴 Max reconnection attempts reached. Please check the WebSocket server.")
            
    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            while self.connected and self.running:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    self.message_queue.put(data)
                    await self.process_message(data)
                except websockets.exceptions.ConnectionClosed:
                    print("🟡 WebSocket connection closed")
                    self.connected = False
                    await self.handle_reconnection()
                    break
                except Exception as e:
                    print(f"🟡 Error processing message: {e}")
                    
        except Exception as e:
            print(f"🔴 Error in message listener: {e}")
            self.connected = False
            
    async def process_message(self, data: Dict):
        """Process incoming WebSocket messages"""
        try:
            message_type = data.get("type")
            
            if message_type == "fraud_alert":
                alert_data = {
                    "timestamp": datetime.now(),
                    "transaction_id": data.get("transaction_id"),
                    "risk_score": data.get("risk_score", 0),
                    "message": data.get("message", "Fraud Alert"),
                    "details": data.get("details", {})
                }
                
                # Call all alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_data)
                    except Exception as e:
                        print(f"🟡 Error in alert callback: {e}")
                        
            elif message_type == "transaction_update":
                transaction_data = data.get("transaction", {})
                
                # Call all transaction callbacks
                for callback in self.transaction_callbacks:
                    try:
                        callback(transaction_data)
                    except Exception as e:
                        print(f"🟡 Error in transaction callback: {e}")
                        
            elif message_type == "system_status":
                print(f"📊 System Status: {data.get('status', 'Unknown')}")
                
        except Exception as e:
            print(f"🟡 Error processing message: {e}")
            
    async def send_message(self, message_type: str, data: Dict):
        """Send a message to the WebSocket server"""
        if self.connected and self.websocket:
            try:
                message = json.dumps({
                    "type": message_type,
                    **data
                })
                await self.websocket.send(message)
            except Exception as e:
                print(f"🟡 Error sending message: {e}")
                
    async def request_test_alert(self):
        """Request a test alert from the server"""
        await self.send_message("test_alert", {})
        
    def start(self):
        """Start the WebSocket client in a separate thread"""
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self.run_async_client, daemon=True)
            thread.start()
            print("🟢 WebSocket client started")
            
    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        self.connected = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        print("🔴 WebSocket client stopped")
        
    def run_async_client(self):
        """Run the async client in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.connect())
        except Exception as e:
            print(f"🔴 Error running WebSocket client: {e}")
        finally:
            loop.close()
            
    def get_latest_messages(self, count: int = 10) -> List[Dict]:
        """Get the latest messages from the queue"""
        messages = []
        try:
            for _ in range(count):
                message = self.message_queue.get_nowait()
                messages.append(message)
        except Empty:
            pass
        return messages

def create_sample_alert():
    """Create a sample fraud alert for testing"""
    return {
        "timestamp": datetime.now(),
        "transaction_id": f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "risk_score": np.random.uniform(0.8, 0.99),
        "message": "High-risk transaction detected",
        "details": {
            "merchant": "Test Merchant",
            "amount": np.random.uniform(100, 1000),
            "country": "US",
            "reason": "Unusual spending pattern"
        }
    }

def main():
    """Test the WebSocket dashboard client"""
    
    def on_alert_received(alert_data):
        print(f"🚨 ALERT RECEIVED: {alert_data['message']}")
        print(f"   Transaction ID: {alert_data['transaction_id']}")
        print(f"   Risk Score: {alert_data['risk_score']:.1%}")
        print(f"   Time: {alert_data['timestamp']}")
        
    def on_transaction_received(transaction_data):
        print(f"📊 TRANSACTION UPDATE: {transaction_data.get('transaction_id', 'Unknown')}")
        print(f"   Amount: ${transaction_data.get('amount', 0):.2f}")
        print(f"   Merchant: {transaction_data.get('merchant', 'Unknown')}")
        
    # Create and start client
    client = WebSocketDashboardClient()
    client.add_alert_callback(on_alert_received)
    client.add_transaction_callback(on_transaction_received)
    
    print("🟢 Starting WebSocket dashboard client...")
    client.start()
    
    # Test the connection
    time.sleep(2)
    
    # Request a test alert
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(client.request_test_alert())
        print("🧪 Test alert requested")
    except Exception as e:
        print(f"🟡 Error requesting test alert: {e}")
    
    # Keep the client running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🟡 Stopping client...")
        client.stop()

if __name__ == "__main__":
    main()