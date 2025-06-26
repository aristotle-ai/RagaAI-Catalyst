#
# from locust import User, task, between, constant_throughput
# from multi_llm_test import run_agentic_task  # Replace with your actual import
# # import debugpy
# # debugpy.listen(5678)
# # debugpy.wait_for_client()
# class AgentUser(User):
#     # wait_time = between(5, 10)
#     wait_time = constant_throughput(10)
#
#     count = 0
#     @task
#     def call_agent(self):
#         result = run_agentic_task()
#         # Optionally assert or log
#         self.count += 1
#         print(f"Call {self.count} completed")
#         assert result is not None


from locust import HttpUser, task, constant_throughput
from multi_llm_test import run_agentic_task  # <-- Replace with actual import path to your main script

class AgentUser(HttpUser):
    # Send 10 traces per second from each user
    wait_time = constant_throughput(10)

    @task
    def call_agent(self):
        try:
            result, context = run_agentic_task()
            if result:
                print("✅ Trace sent successfully")
            else:
                print("⚠️ No result returned")
        except Exception as e:
            print(f"❌ Trace failed: {e}")

