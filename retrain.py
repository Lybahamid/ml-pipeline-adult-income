from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import logging
import tqdm
import subprocess

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.data'):
            logging.info(f"New data file detected: {event.src_path}")
            logging.info("Retraining pipeline...")
            os.system('python main.py')

if __name__ == "__main__":
    logging.debug("Starting directory monitoring for retraining")
    observer = Observer()
    observer.schedule(DataHandler(), path='data', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.debug("Stopping directory monitoring")
        observer.stop()
    observer.join()