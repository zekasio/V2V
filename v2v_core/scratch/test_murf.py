import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("MURFDUB_API_KEY")
job_id = "DJ13i5vrgh1RM" # the job we just ran

from murf import MurfDub as MurfDubSDK
client = MurfDubSDK(api_key=api_key)

print("Fetching job status...")
try:
    response = client.dubbing.jobs.get_status(job_id=job_id)
    print("Type of response:", type(response))
    print("response:", response)
    print("hasattr __dict__:", hasattr(response, "__dict__"))
    if hasattr(response, "__dict__"):
        print("response.__dict__ type:", type(response.__dict__))
        print("response.__dict__:", response.__dict__)
except Exception as e:
    print("Error:", e)
