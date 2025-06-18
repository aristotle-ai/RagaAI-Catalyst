from locust import HttpUser, task, between, events
import json
import logging

# Enable verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NanoEndpointUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:5001"

    def on_start(self):
        """On start, verify the server is accessible"""
        response = self.client.get("/")
        if response.status_code == 200:
            logger.info("Successfully connected to the server")
        else:
            logger.error(f"Failed to connect to server. Status code: {response.status_code}")

    @task
    def test_nano_endpoint(self):
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        data = {
            "message": "Provide specific title in 20 words about the doc"
        }
        
        try:
            with self.client.post(
                "/nano",
                json=data,
                headers=headers,
                catch_response=True,
                name="nano_endpoint"
            ) as response:
                # Log the full response for debugging
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {response.headers}")
                logger.info(f"Response content: {response.text}")
                
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(
                        f"Status: {response.status_code}, "
                        f"Response: {response.text[:200]}"
                    )
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            response.failure(f"Request failed: {str(e)}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("Test is starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Test is stopping...")
