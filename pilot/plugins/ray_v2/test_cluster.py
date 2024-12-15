import subprocess
import unittest
from unittest.mock import patch

from pilot.plugins.ray_v2.cluster import RayManager

TMP_DIR = "/tmp"

class TestCluster(unittest.TestCase):    
    def test_start_scheduler(self):        
        cluster = RayManager(TMP_DIR)

        # Mock the subprocess.Popen call
        with patch('subprocess.Popen') as mock_popen:
            # Mock the stdout and stderr attributes of the Popen object
            mock_popen.return_value.stdout.read.return_value.decode.return_value = "Scheduler details"

            # Call the start_scheduler method
            cluster.start_scheduler()

            # Assert that Popen was called with the correct arguments
            mock_popen.assert_called_once_with(["ray", "start", "--head"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Assert that the scheduler details were written to the file
            with open("scheduler_details.txt", "r") as file:
                self.assertEqual(file.read(), "Scheduler details")



if __name__ == '__main__':
    unittest.main()