import threading
import psutil
import datetime
import json
import os

class DebugUtility:
    """Utility for capturing system and thread state to a file."""
    @staticmethod
    def dump_state_to_file():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_state_{timestamp}.json"
        data = {}
        # Gather thread information
        threads = []
        for t in threading.enumerate():
            threads.append({
                "name": t.name,
                "ident": t.ident,
                "is_alive": t.is_alive(),
                "daemon": t.daemon
            })
        data["threads"] = threads
        # Gather process metrics
        proc = psutil.Process()
        data["process_cpu_percent"] = proc.cpu_percent(interval=1.0)
        data["process_memory_info"] = proc.memory_info()._asdict()
        # Write to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return os.path.abspath(filename)