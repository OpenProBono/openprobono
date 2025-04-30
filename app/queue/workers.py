"""Workers for processing jobs from the queue."""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict
import os

from app.db import (
    load_bot,
    set_session_to_bot,
    update_run_session_job_status,
)
from app.logger import setup_logger
from app.models import (
    ChatRequest,
    JobStatus,
    RunSessionJob,
    get_uuid_id,
    User,
)
from app.queue.client import (
    RUN_QUEUE,
    get_queue_client,
)
from app.main import process_chat

logger = setup_logger()

# Number of retries for failed job items
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 5


async def process_run_job(job: RunSessionJob) -> bool:
    """Process a run job.

    Parameters
    ----------
    job_data : RunSessionJob
        Job data from the queue

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Check if bot exists and user has access
        bot = load_bot(job.bot_id)
        if not bot:
            logger.error(f"Bot {job.bot_id} not found for run {job.run_id}")
            update_run_session_job_status(job.id, JobStatus.FAILED, error_message=f"Bot {job.bot_id} not found")
            return False
        
        # Create a new session for this input-bot pair
        session_id = get_uuid_id()
        set_session_to_bot(session_id, job.bot_id)
        
        # Initialize the session with the input
        cr = ChatRequest(
            history=[{"role": "user", "content": job.input_text}],
            bot_id=job.bot_id,
            session_id=session_id,
            user=job.user,
        )
        
        # Call the bot to get the output
        response = process_chat(cr, job.input_text)
        output_text = response.get("output", "Error: No output generated")
        
        job.output_text = output_text
        
        # Update the job status
        update_run_session_job_status(job.id, JobStatus.COMPLETED, output_text=output_text, session_id=session_id)
        logger.info(f"Run job {job.id} processed successfully")
        return True
    except Exception as e:
        error_message = f"Error processing run job {job.id}: {str(e)}"
        logger.error(error_message)
        update_run_session_job_status(job.id, JobStatus.FAILED, error_message=error_message)
        return False


async def start_workers_async(worker_type: str, input_workers: int, run_workers: int) -> None:
    """Run the specified workers asynchronously.
    
    Parameters
    ----------
    worker_type : str
        Type of worker(s) to run: "input", "run", or "both"
    """
    client = await get_queue_client()
    logger.info(f"Starting workers: {worker_type}")
    
    tasks = []
    
    # if worker_type in ["input", "both"]:
    #     # For input worker
    #     tasks.append(client.consume_messages(
    #         INPUT_GENERATION_QUEUE,
    #         process_input_generation_job,
    #         prefetch_count=input_workers
    #     ))
    
    if worker_type in ["run", "both"]:
        # For run worker
        async def run_job_wrapper(job_data: RunSessionJob) -> bool:
            """Wrapper for process_run_job that handles both dict and RunSessionJob inputs.
            
            Parameters
            ----------
            job_data : RunSessionJob
                Job data from the queue
                
            Returns
            -------
            bool
                Result of processing
            """
            # If job_data is already a RunSessionJob object, use it directly
            if isinstance(job_data, RunSessionJob):
                return await process_run_job(job_data)
            
            # Otherwise, convert dict to RunSessionJob
            try:
                # Check if we need to manually process the user data
                if 'user' in job_data and isinstance(job_data['user'], dict):
                    # Recreate User object from its dictionary representation
                    job_data['user'] = User(**job_data['user'])
                
                job = RunSessionJob(**job_data)
                return await process_run_job(job)
            except Exception as e:
                logger.error(f"Error converting job data to RunSessionJob: {e}")
                # Try to update job status if possible
                if 'id' in job_data:
                    update_run_session_job_status(
                        job_data['id'], 
                        JobStatus.FAILED, 
                        error_message=f"Error deserializing job: {str(e)}"
                    )
                return False
        
        tasks.append(client.consume_messages(
            RUN_QUEUE,
            run_job_wrapper,
            prefetch_count=run_workers
        ))
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logger.info("Worker module loaded - starting up...")
    logger.debug("Worker module loaded - starting up...")
    print(f"Worker starting up at {time.strftime('%H:%M:%S')}", flush=True)
    sys.stdout.flush()
    print(f"RABBITMQ settings: host={os.environ.get('RABBITMQ_HOST', 'localhost')}", flush=True)
    sys.stdout.flush()
    logger.info("Starting workers")
    import argparse
    
    parser = argparse.ArgumentParser(description="Start queue workers")
    parser.add_argument(
        "--worker", 
        choices=["input", "eval", "both"], 
        default="both",
        help="Worker type to start"
    )
    parser.add_argument(
        "--input-workers",
        type=int,
        default=1,
        help="Number of input generation workers to start"
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of evaluation workers to start"
    )
    args = parser.parse_args()
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Define task to run
    task = start_workers_async(args.worker, args.input_workers, args.eval_workers)
    
    try:
        # Run the worker task - the run_workers_async function has been modified to block indefinitely
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        # Handle ctrl+c gracefully
        logger.info("Keyboard interrupt received, shutting down workers")
        # Cancel the task
        task.cancel()
        try:
            # Wait for the task to be canceled
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
    except Exception as e:
        logger.error(f"Worker failed with error: {e}")
    finally:
        # Close the event loop
        loop.close() 