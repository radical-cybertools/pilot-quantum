import uuid
import os


class PilotManager:
    def __init__(self, working_directory, framework_type):
        self.working_directory = working_directory
        self.job_id = uuid.uuid1().__str__()
        self.pilot_working_directory = os.path.join(self.working_directory, self.job_id)
        self.agent_output_log =
        self.agent_error_log =

    def create_pilot(self):
        pass

    def submit_pilot(self):
        pass

    def get_pilot_status(self):
        pass

    def cancel_pilot(self):
        pass
