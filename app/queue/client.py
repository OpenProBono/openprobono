"""RabbitMQ client implementation for job queues."""
from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Callable, Dict, Optional, Coroutine

import aio_pika
from aio_pika.exceptions import AMQPConnectionError

from app.logger import setup_logger
from app.models import RunSessionJob

logger = setup_logger()

# Queue names
INPUT_GENERATION_QUEUE = "input_generation"
RUN_QUEUE = "run"
DEAD_LETTER_QUEUE = "dead_letter"

# RabbitMQ connection settings - these would typically come from environment variables
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


class AsyncQueueClient:
    """Asynchronous client for interacting with RabbitMQ message queues."""

    def __init__(self):
        """Initialize the async queue client."""
        self.connection = None
        self.channel = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection and setup queues asynchronously."""
        if not self._initialized:
            await self._connect()
            await self._setup_queues()
            self._initialized = True

    def _get_connection_url(self) -> str:
        """Get connection URL for RabbitMQ.
        
        Returns
        -------
        str
            Connection URL for RabbitMQ
        """
        return f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/{RABBITMQ_VHOST}"

    async def _connect(self) -> None:
        """Connect to RabbitMQ with retry logic."""
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                self.connection = await aio_pika.connect_robust(
                    self._get_connection_url(),
                    timeout=30,
                )
                self.channel = await self.connection.channel()
                logger.info("Connected to RabbitMQ")
                break
            except AMQPConnectionError as e:
                retry_count += 1
                logger.warning(f"Failed to connect to RabbitMQ (attempt {retry_count}/{MAX_RETRIES}): {e}")
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Failed to connect to RabbitMQ after {MAX_RETRIES} attempts")
                    raise
                await asyncio.sleep(RETRY_DELAY)

    async def _setup_queues(self) -> None:
        """Set up the queues and exchanges needed for the application."""
        # Declare dead letter exchange
        await self.channel.declare_exchange(
            name="dlx", 
            type=aio_pika.ExchangeType.DIRECT,
            durable=True
        )
        
        # Declare dead letter queue
        dead_letter_queue = await self.channel.declare_queue(
            name=DEAD_LETTER_QUEUE,
            durable=True
        )
        
        # Get dead letter exchange
        dlx = await self.channel.get_exchange(name="dlx")
        
        # Bind dead letter queue to dead letter exchange
        await dead_letter_queue.bind(
            exchange=dlx,
            routing_key="dead-letter"
        )
        
        # Declare main queues with dead letter configuration
        for queue_name in [INPUT_GENERATION_QUEUE, RUN_QUEUE]:
            await self.channel.declare_queue(
                name=queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": "dead-letter"
                }
            )
            logger.info(f"Declared queue: {queue_name}")

    async def reconnect_if_needed(self) -> None:
        """Reconnect to RabbitMQ if the connection is closed."""
        if not self.connection or self.connection.is_closed:
            logger.info("Connection is closed, reconnecting...")
            await self._connect()
            await self._setup_queues()

    async def publish_message(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """Publish a message to a queue asynchronously.
        
        Parameters
        ----------
        queue_name : str
            Name of the queue to publish to
        message : Dict[str, Any]
            Message to publish (will be serialized to JSON)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            await self.reconnect_if_needed()
            
            # Get the queue
            queue = await self.channel.declare_queue(
                name=queue_name, 
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": "dead-letter"
                }
            )
            
            # Get the default exchange
            exchange = self.channel.default_exchange
            
            # Create a message
            message_body = json.dumps(message).encode()
            aio_message = aio_pika.Message(
                body=message_body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json"
            )
            
            # Publish the message
            await exchange.publish(
                message=aio_message,
                routing_key=queue_name
            )
            
            logger.info(f"Published message to {queue_name}")
            return True
        except AMQPConnectionError as e:
            logger.error(f"Failed to publish message to {queue_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing to {queue_name}: {e}")
            return False

    async def consume_messages(
        self, 
        queue_name: str, 
        callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, bool]],
        prefetch_count: int = 1
    ) -> None:
        """Consume messages from a queue asynchronously.
        
        Parameters
        ----------
        queue_name : str
            Name of the queue to consume from
        callback : Callable[[Dict[str, Any]], Coroutine[Any, Any, bool]]
            Async callback function that receives the message and returns True if successfully processed
        prefetch_count : int, optional
            Number of messages to prefetch, by default 1
        """
        async def _process_message(message: aio_pika.IncomingMessage) -> None:
            try:
                # Parse message body
                body = message.body.decode()
                data = json.loads(body)
                
                # Convert to RunSessionJob if appropriate
                if queue_name == RUN_QUEUE:
                    try:
                        from app.models import RunSessionJob, User
                        # Check if we need to manually process the user data
                        if 'user' in data and isinstance(data['user'], dict):
                            # Recreate User object from its dictionary representation
                            data['user'] = User(**data['user'])
                        job_data = RunSessionJob(**data)
                    except Exception as e:
                        logger.error(f"Error deserializing RunSessionJob: {e}")
                        # Continue with raw data in this case
                        job_data = data
                else:
                    job_data = data
                
                # Track retry count in message headers
                retry_count = 0
                if message.headers and 'x-retry-count' in message.headers:
                    retry_count = message.headers['x-retry-count']
                
                # Call the callback
                success = await callback(job_data)
                
                if success:
                    # Acknowledge message
                    logger.info(f"Successfully processed message from {queue_name}")
                    await message.ack()
                else:
                    # Check retry limit
                    if retry_count >= MAX_RETRIES:
                        logger.warning(f"Message failed after {retry_count} retries, sending to dead letter queue")
                        await message.reject(requeue=False)
                    else:
                        # Increment retry count and republish
                        logger.info(f"Message processing failed, retry {retry_count+1}/{MAX_RETRIES}")
                        
                        # Republish with updated retry count
                        new_message = aio_pika.Message(
                            body=message.body,
                            headers={'x-retry-count': retry_count + 1},
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            content_type="application/json"
                        )
                        
                        # Publish to same queue after delay
                        await asyncio.sleep(RETRY_DELAY)
                        await self.channel.default_exchange.publish(
                            new_message,
                            routing_key=queue_name
                        )
                        
                        # Acknowledge the original message
                        await message.ack()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Reject and don't requeue - goes to dead letter queue
                await message.reject(requeue=False)
        
        try:
            await self.reconnect_if_needed()
            
            # Set prefetch count
            await self.channel.set_qos(prefetch_count=prefetch_count)
            
            # Get the queue
            queue = await self.channel.declare_queue(
                name=queue_name, 
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": "dead-letter"
                }
            )
            
            # Start consuming
            await queue.consume(_process_message)
            
            logger.info(f"Started consuming from {queue_name}")
            
            # Keep the consumer running
            while True:
                await asyncio.sleep(1)
                
        except AMQPConnectionError as e:
            logger.error(f"Connection error while consuming from {queue_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error consuming from {queue_name}: {e}")
            raise

    async def close(self) -> None:
        """Close the connection to RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Closed RabbitMQ connection")


# Singleton instance
_client: Optional[AsyncQueueClient] = None
_client_lock = asyncio.Lock()


async def get_queue_client() -> AsyncQueueClient:
    """Get the singleton queue client instance asynchronously.
    
    Returns
    -------
    AsyncQueueClient
        Singleton queue client instance
    """
    global _client
    
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = AsyncQueueClient()
                await _client.initialize()
    
    return _client


def get_queue_client_sync() -> AsyncQueueClient:
    """Get the singleton queue client instance synchronously.
    
    Returns
    -------
    AsyncQueueClient
        Singleton queue client instance
    """
    global _client
    
    if _client is None:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If loop is running, we should use a synchronous approach to avoid nesting loops
            # Create client manually and initialize it in the background
            _client = AsyncQueueClient()
            # Schedule initialization task to run in the background
            asyncio.create_task(_client.initialize())
        else:
            # If loop is not running, we can use it directly
            _client = loop.run_until_complete(get_queue_client())
    
    return _client


async def _publish_input_generation_job_async(job_data: Dict[str, Any]) -> bool:
    """Publish an input generation job to the queue asynchronously.
    
    Parameters
    ----------
    job_data : Dict[str, Any]
        Job data to publish
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    client = get_queue_client_sync()
    return await client.publish_message(INPUT_GENERATION_QUEUE, job_data)

async def _publish_run_job_async(job_data: RunSessionJob) -> bool:
    """Publish an evaluation job to the queue asynchronously.
    
    Parameters
    ----------
    job_data : RunSessionJob
        Job data to publish
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    client = get_queue_client_sync()
    return await client.publish_message(RUN_QUEUE, job_data)


# Synchronous wrapper functions for API endpoints
def publish_input_generation_job(job_data: Dict[str, Any]) -> bool:
    """Synchronous wrapper for publishing an input generation job.
    
    Parameters
    ----------
    job_data : Dict[str, Any]
        Job data to publish
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Try to use an existing event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Check if we're already running in the event loop
        if loop.is_running():
            logger.warning("Event loop is already running, creating a new one")
            loop = asyncio.new_event_loop()
        
        return loop.run_until_complete(_publish_input_generation_job_async(job_data))
    except Exception as e:
        logger.error(f"Failed to publish input generation job: {e}")
        return False

def publish_run_session_job(job: RunSessionJob) -> bool:
    """Publish a run session job to the queue
    
    Parameters
    ----------
    job : RunSessionJob
        The run session job to publish
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Try to use an existing event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Check if we're already running in the event loop
        if loop.is_running():
            logger.warning("Event loop is already running, creating a new one")
            loop = asyncio.new_event_loop()
        
        return loop.run_until_complete(_publish_run_job_async(job.model_dump()))
    except Exception as e:
        logger.error(f"Failed to publish evaluation job: {e}")
        # Instead of failing, return False to indicate failure but allow the API to continue
        return False