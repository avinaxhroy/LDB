# app/collectors/artist_queue.py

import threading
import queue
import time
from typing import Tuple
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.collectors.artist_catalog import artist_catalog_collector
import logging

logger = logging.getLogger(__name__)

# Create a queue for artist processing
artist_queue = queue.Queue()


def process_artist_worker():
    """Worker thread to process artists from the queue"""
    logger.info("Starting artist catalog worker thread")
    while True:
        try:
            # Get artist from queue (blocks until item is available)
            artist_name, spotify_id = artist_queue.get()

            # Create new DB session (thread-safe)
            db = SessionLocal()
            try:
                # Process the artist's catalog
                logger.info(f"Processing catalog for artist: {artist_name}")
                new_tracks = artist_catalog_collector.process_catalog(db, artist_name, spotify_id)
                logger.info(f"Added {new_tracks} new tracks for artist {artist_name}")
            except Exception as e:
                logger.error(f"Error processing artist {artist_name}: {str(e)}")
            finally:
                db.close()

            # Mark task as done
            artist_queue.task_done()

            # Sleep to prevent overloading
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in artist processing worker: {str(e)}")
            time.sleep(5)  # Sleep longer after an error


def queue_artist_for_processing(artist_name: str, spotify_id: str = None):
    """Add artist to processing queue"""
    if artist_name and artist_name.lower() not in ["unknown", "unknown artist", "various artists"]:
        logger.info(f"Queuing artist for catalog processing: {artist_name}")
        artist_queue.put((artist_name, spotify_id))


# Start worker thread
worker_thread = threading.Thread(target=process_artist_worker, daemon=True)
worker_thread.start()
