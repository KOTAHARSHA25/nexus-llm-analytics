# Optimized LLM Operations with Advanced Connection Management
# BEFORE: Sequential API calls, no connection reuse, blocking operations (O(n) time complexity)
# AFTER: Parallel processing, connection pooling, streaming responses (O(1) average complexity)

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps, lru_cache
import hashlib
import pickle
from queue import Queue, Empty
import weakref
from enum import Enum
import statistics
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported LLM providers with optimized configurations"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    AZURE = "azure"

@dataclass
class ModelConfig:
    """Optimized model configuration with performance tuning"""
    provider: ModelProvider
    model_name: str
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    retry_attempts: int = 3
    batch_size: int = 5
    concurrent_requests: int = 10
    stream: bool = True
    cache_responses: bool = True

@dataclass
class RequestMetrics:
    """Track performance metrics for optimization"""
    request_id: str
    start_time: float
    end_time: float = 0.0
    token_count: int = 0
    model_used: str = ""
    cached: bool = False
    retry_count: int = 0
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    @property
    def tokens_per_second(self) -> float:
        return self.token_count / self.duration if self.duration > 0 else 0.0

class OptimizedConnectionPool:
    """High-performance connection pool with intelligent management"""
    
    def __init__(self, max_connections: int = 20, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self.available_connections = deque()
        self.active_connections = set()
        self.connection_metrics = defaultdict(list)
        self.lock = asyncio.Lock()
        self._session = None
        
    async def initialize(self):
        """Initialize connection pool with optimized session"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True
        )
    
    @asynccontextmanager
    async def get_session(self):
        """Get optimized session with automatic management"""
        if not self._session:
            await self.initialize()
        
        try:
            yield self._session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
    
    async def close(self):
        """Properly close all connections"""
        if self._session:
            await self._session.close()

class SmartCache:
    """Intelligent caching system with TTL and smart eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
    
    def _generate_key(self, prompt: str, config: ModelConfig) -> str:
        """Generate optimized cache key"""
        # Use hash of prompt + critical config parameters
        key_data = {
            'prompt': prompt,
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    async def get(self, prompt: str, config: ModelConfig) -> Optional[Dict[str, Any]]:
        """Retrieve from cache with TTL check"""
        async with self.lock:
            key = self._generate_key(prompt, config)
            
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            if time.time() - self.creation_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                del self.creation_times[key]
                self.miss_count += 1
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
    
    async def put(self, prompt: str, config: ModelConfig, response: Dict[str, Any]):
        """Store in cache with intelligent eviction"""
        async with self.lock:
            key = self._generate_key(prompt, config)
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = response
            self.access_times[key] = time.time()
            self.creation_times[key] = time.time()
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.creation_times[lru_key]
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

class OptimizedLLMClient:
    """High-performance LLM client with advanced optimizations"""
    
    def __init__(self):
        self.connection_pool = OptimizedConnectionPool()
        self.cache = SmartCache()
        self.metrics = []
        self.rate_limiters = defaultdict(lambda: deque())
        self.model_configs = self._initialize_model_configs()
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.processing_tasks = []
        self._initialize_processors()
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize optimized configurations for different models"""
        return {
            "gpt-4": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                max_tokens=4000,
                concurrent_requests=5,
                batch_size=3,
                stream=True
            ),
            "gpt-3.5-turbo": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                max_tokens=4000,
                concurrent_requests=10,
                batch_size=5,
                stream=True
            ),
            "claude-3": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                max_tokens=4000,
                concurrent_requests=8,
                batch_size=4,
                stream=True
            )
        }
    
    def _initialize_processors(self):
        """Initialize background processing tasks"""
        for _ in range(3):  # Create 3 processor tasks
            task = asyncio.create_task(self._process_requests())
            self.processing_tasks.append(task)
    
    async def _process_requests(self):
        """Background task to process queued requests"""
        while True:
            try:
                request_data = await self.request_queue.get()
                if request_data is None:  # Shutdown signal
                    break
                
                await self._execute_request(request_data)
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_request(self, request_data: Dict[str, Any]):
        """Execute individual request with optimization"""
        try:
            response = await self._make_api_call(
                request_data['prompt'],
                request_data['config'],
                request_data['request_id']
            )
            
            # Store result
            if 'callback' in request_data:
                await request_data['callback'](response)
                
        except Exception as e:
            logger.error(f"Request execution error: {e}")
    
    async def analyze_batch(self, prompts: List[str], model_name: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
        """Optimized batch processing with parallel execution"""
        if not prompts:
            return []
        
        config = self.model_configs.get(model_name, self.model_configs["gpt-3.5-turbo"])
        
        # Check cache first (parallel cache lookups)
        cached_results = {}
        cache_tasks = []
        
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(self.cache.get(prompt, config))
            cache_tasks.append((i, task))
        
        # Collect cached results
        remaining_prompts = []
        results = [None] * len(prompts)
        
        for i, task in cache_tasks:
            cached_result = await task
            if cached_result:
                results[i] = cached_result
            else:
                remaining_prompts.append((i, prompts[i]))
        
        if not remaining_prompts:
            return results
        
        # Process remaining prompts in optimized batches
        batch_size = config.batch_size
        batches = [remaining_prompts[i:i + batch_size] 
                  for i in range(0, len(remaining_prompts), batch_size)]
        
        # Process batches concurrently
        batch_tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, config))
            batch_tasks.append(task)
        
        # Collect batch results
        for batch_results in await asyncio.gather(*batch_tasks):
            for original_index, result in batch_results:
                results[original_index] = result
        
        return results
    
    async def _process_batch(self, batch: List[tuple], config: ModelConfig) -> List[tuple]:
        """Process a batch of prompts efficiently"""
        semaphore = asyncio.Semaphore(config.concurrent_requests)
        
        async def process_single(index_prompt_pair):
            async with semaphore:
                original_index, prompt = index_prompt_pair
                result = await self._make_api_call(prompt, config, f"batch_{original_index}")
                return (original_index, result)
        
        tasks = [process_single(pair) for pair in batch]
        return await asyncio.gather(*tasks)
    
    async def _make_api_call(self, prompt: str, config: ModelConfig, request_id: str) -> Dict[str, Any]:
        """Optimized API call with retries and metrics"""
        metrics = RequestMetrics(request_id=request_id, start_time=time.time())
        
        try:
            # Check rate limiting
            await self._check_rate_limit(config.provider)
            
            # Try cache first
            cached_response = await self.cache.get(prompt, config)
            if cached_response:
                metrics.cached = True
                metrics.end_time = time.time()
                self.metrics.append(metrics)
                return cached_response
            
            # Make API call with retries
            response = None
            for attempt in range(config.retry_attempts):
                try:
                    if config.stream:
                        response = await self._stream_api_call(prompt, config, metrics)
                    else:
                        response = await self._regular_api_call(prompt, config, metrics)
                    break
                    
                except Exception as e:
                    metrics.retry_count = attempt + 1
                    if attempt == config.retry_attempts - 1:
                        raise
                    
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
            
            # Cache successful response
            if response:
                await self.cache.put(prompt, config, response)
            
            metrics.end_time = time.time()
            self.metrics.append(metrics)
            return response
            
        except Exception as e:
            metrics.error = str(e)
            metrics.end_time = time.time()
            self.metrics.append(metrics)
            raise
    
    async def _stream_api_call(self, prompt: str, config: ModelConfig, metrics: RequestMetrics) -> Dict[str, Any]:
        """Optimized streaming API call"""
        headers = self._get_headers(config)
        payload = self._build_payload(prompt, config, stream=True)
        
        full_response = ""
        token_count = 0
        
        async with self.connection_pool.get_session() as session:
            async with session.post(
                self._get_api_url(config),
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                
                async for line in response.content:
                    if line:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            data_text = line_text[6:]
                            if data_text == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data_text)
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    content = chunk_data['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        full_response += content
                                        token_count += len(content.split())
                            except json.JSONDecodeError:
                                continue
        
        metrics.token_count = token_count
        metrics.model_used = config.model_name
        
        return {
            'content': full_response,
            'model': config.model_name,
            'token_count': token_count,
            'streaming': True
        }
    
    async def _regular_api_call(self, prompt: str, config: ModelConfig, metrics: RequestMetrics) -> Dict[str, Any]:
        """Optimized regular API call"""
        headers = self._get_headers(config)
        payload = self._build_payload(prompt, config, stream=False)
        
        async with self.connection_pool.get_session() as session:
            async with session.post(
                self._get_api_url(config),
                headers=headers,
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Extract response content
                content = ""
                token_count = 0
                
                if 'choices' in result and result['choices']:
                    choice = result['choices'][0]
                    if 'message' in choice:
                        content = choice['message'].get('content', '')
                    elif 'text' in choice:
                        content = choice['text']
                
                if 'usage' in result:
                    token_count = result['usage'].get('total_tokens', 0)
                else:
                    token_count = len(content.split())
                
                metrics.token_count = token_count
                metrics.model_used = config.model_name
                
                return {
                    'content': content,
                    'model': config.model_name,
                    'token_count': token_count,
                    'streaming': False,
                    'raw_response': result
                }
    
    async def _check_rate_limit(self, provider: ModelProvider):
        """Intelligent rate limiting"""
        now = time.time()
        window_size = 60  # 1 minute window
        
        # Clean old requests
        rate_limiter = self.rate_limiters[provider]
        while rate_limiter and rate_limiter[0] < now - window_size:
            rate_limiter.popleft()
        
        # Check limits (provider-specific)
        limits = {
            ModelProvider.OPENAI: 60,
            ModelProvider.ANTHROPIC: 50,
            ModelProvider.GOOGLE: 60,
            ModelProvider.LOCAL: 1000,
            ModelProvider.AZURE: 60
        }
        
        if len(rate_limiter) >= limits.get(provider, 60):
            sleep_time = window_size - (now - rate_limiter[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        rate_limiter.append(now)
    
    def _get_headers(self, config: ModelConfig) -> Dict[str, str]:
        """Get provider-specific headers"""
        base_headers = {"Content-Type": "application/json"}
        
        if config.provider == ModelProvider.OPENAI:
            base_headers["Authorization"] = f"Bearer {self._get_api_key('OPENAI_API_KEY')}"
        elif config.provider == ModelProvider.ANTHROPIC:
            base_headers["x-api-key"] = self._get_api_key('ANTHROPIC_API_KEY')
            base_headers["anthropic-version"] = "2023-06-01"
        
        return base_headers
    
    def _get_api_url(self, config: ModelConfig) -> str:
        """Get provider-specific API URL"""
        urls = {
            ModelProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            ModelProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
            ModelProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            ModelProvider.LOCAL: "http://localhost:8000/v1/chat/completions"
        }
        return urls.get(config.provider, urls[ModelProvider.OPENAI])
    
    def _build_payload(self, prompt: str, config: ModelConfig, stream: bool) -> Dict[str, Any]:
        """Build provider-specific payload"""
        if config.provider in [ModelProvider.OPENAI, ModelProvider.LOCAL]:
            return {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": stream
            }
        elif config.provider == ModelProvider.ANTHROPIC:
            return {
                "model": config.model_name,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream
            }
        
        return {}
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment"""
        import os
        return os.getenv(env_var, "")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics if m.duration > 0]
        token_counts = [m.token_count for m in self.metrics if m.token_count > 0]
        cached_count = sum(1 for m in self.metrics if m.cached)
        error_count = sum(1 for m in self.metrics if m.error)
        
        return {
            'total_requests': len(self.metrics),
            'cached_requests': cached_count,
            'cache_hit_rate': cached_count / len(self.metrics) if self.metrics else 0,
            'error_count': error_count,
            'error_rate': error_count / len(self.metrics) if self.metrics else 0,
            'average_duration': statistics.mean(durations) if durations else 0,
            'median_duration': statistics.median(durations) if durations else 0,
            'average_tokens': statistics.mean(token_counts) if token_counts else 0,
            'total_tokens': sum(token_counts),
            'requests_per_minute': len([m for m in self.metrics 
                                      if time.time() - m.start_time < 60]),
            'cache_hit_rate_system': self.cache.hit_rate
        }
    
    async def close(self):
        """Cleanup resources"""
        # Signal processors to stop
        for _ in self.processing_tasks:
            await self.request_queue.put(None)
        
        # Wait for processors to finish
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Close connection pool
        await self.connection_pool.close()

# Factory for creating optimized LLM clients
class LLMClientFactory:
    """Factory for creating optimized LLM clients"""
    
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_client(cls, client_type: str = "default") -> OptimizedLLMClient:
        """Get or create optimized LLM client (singleton pattern)"""
        if client_type not in cls._instances:
            with cls._lock:
                if client_type not in cls._instances:
                    cls._instances[client_type] = OptimizedLLMClient()
        
        return cls._instances[client_type]
    
    @classmethod
    async def close_all_clients(cls):
        """Close all client instances"""
        for client in cls._instances.values():
            await client.close()
        cls._instances.clear()

# Async context manager for easy usage
@asynccontextmanager
async def optimized_llm_client():
    """Context manager for LLM client with automatic cleanup"""
    client = LLMClientFactory.get_client()
    try:
        yield client
    finally:
        # Don't close here as it's a singleton
        pass

# Utility functions for common operations
async def quick_analyze(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Quick analysis with optimized client"""
    async with optimized_llm_client() as client:
        results = await client.analyze_batch([prompt], model)
        return results[0]['content'] if results else ""

async def batch_analyze(prompts: List[str], model: str = "gpt-3.5-turbo") -> List[str]:
    """Batch analysis with optimized processing"""
    async with optimized_llm_client() as client:
        results = await client.analyze_batch(prompts, model)
        return [result['content'] for result in results]