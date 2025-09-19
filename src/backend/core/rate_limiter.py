# Rate Limiting Module for API Protection
# Implements token bucket algorithm for rate limiting

import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import asyncio
from enum import Enum

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.message = message
        self.retry_after = retry_after  # Seconds until next request allowed
        super().__init__(self.message)

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, refill_amount: int = 1):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Time in seconds between refills
            refill_amount: Number of tokens to add per refill
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_amount = refill_amount
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (success, wait_time_if_failed)
        """
        async with self._lock:
            # Refill tokens based on elapsed time
            current_time = time.time()
            elapsed = current_time - self.last_refill
            refills = int(elapsed / self.refill_rate)
            
            if refills > 0:
                self.tokens = min(
                    self.capacity,
                    self.tokens + (refills * self.refill_amount)
                )
                self.last_refill = current_time
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            refills_needed = (tokens_needed + self.refill_amount - 1) // self.refill_amount
            wait_time = refills_needed * self.refill_rate
            
            return False, wait_time

class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize sliding window counter
        
        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests allowed in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> Tuple[bool, float]:
        """
        Check if request is allowed
        
        Returns:
            Tuple of (allowed, wait_time_if_not_allowed)
        """
        async with self._lock:
            current_time = time.time()
            window_start = current_time - self.window_size
            
            # Remove old requests outside window
            self.requests = [t for t in self.requests if t > window_start]
            
            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True, 0
            
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = oldest_request + self.window_size - current_time
            
            return False, wait_time

class RateLimiter:
    """Main rate limiter class with multiple strategies"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        enable_user_limits: bool = True,
        enable_ip_limits: bool = True
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            burst_size: Max burst requests
            strategy: Rate limiting strategy
            enable_user_limits: Enable per-user rate limiting
            enable_ip_limits: Enable per-IP rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.strategy = strategy
        self.enable_user_limits = enable_user_limits
        self.enable_ip_limits = enable_ip_limits
        
        # Storage for different rate limiters
        self.user_limiters: Dict[str, TokenBucket] = {}
        self.ip_limiters: Dict[str, TokenBucket] = {}
        self.global_limiter = self._create_limiter()
        
        # Statistics
        self.stats = defaultdict(lambda: {"allowed": 0, "blocked": 0})
        
        logging.info(f"Rate limiter initialized: {requests_per_minute}/min, {requests_per_hour}/hour")
    
    def _create_limiter(self) -> TokenBucket:
        """Create a rate limiter based on strategy"""
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            # Create token bucket with per-minute rate
            return TokenBucket(
                capacity=self.burst_size,
                refill_rate=60 / self.requests_per_minute,
                refill_amount=1
            )
        else:
            # Default to token bucket for now
            return TokenBucket(
                capacity=self.burst_size,
                refill_rate=60 / self.requests_per_minute,
                refill_amount=1
            )
    
    async def check_rate_limit(
        self,
        identifier: Optional[str] = None,
        ip_address: Optional[str] = None,
        tokens: int = 1
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: User identifier (user_id, api_key, etc.)
            ip_address: IP address of request
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # Check global rate limit
        allowed, wait_time = await self.global_limiter.consume(tokens)
        if not allowed:
            self._update_stats("global", False)
            return False, int(wait_time)
        
        # Check user-specific rate limit
        if self.enable_user_limits and identifier:
            if identifier not in self.user_limiters:
                self.user_limiters[identifier] = self._create_limiter()
            
            allowed, wait_time = await self.user_limiters[identifier].consume(tokens)
            if not allowed:
                self._update_stats(f"user:{identifier}", False)
                return False, int(wait_time)
        
        # Check IP-specific rate limit
        if self.enable_ip_limits and ip_address:
            if ip_address not in self.ip_limiters:
                self.ip_limiters[ip_address] = self._create_limiter()
            
            allowed, wait_time = await self.ip_limiters[ip_address].consume(tokens)
            if not allowed:
                self._update_stats(f"ip:{ip_address}", False)
                return False, int(wait_time)
        
        # Request allowed
        self._update_stats("global", True)
        if identifier:
            self._update_stats(f"user:{identifier}", True)
        if ip_address:
            self._update_stats(f"ip:{ip_address}", True)
        
        return True, None
    
    def _update_stats(self, key: str, allowed: bool):
        """Update statistics"""
        if allowed:
            self.stats[key]["allowed"] += 1
        else:
            self.stats[key]["blocked"] += 1
    
    def get_stats(self) -> Dict:
        """Get rate limiting statistics"""
        return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats.clear()
    
    def cleanup_old_limiters(self, max_age_seconds: int = 3600):
        """Clean up old unused limiters to free memory"""
        # This would need to track last access time for each limiter
        # For now, just clear if too many
        max_limiters = 1000
        
        if len(self.user_limiters) > max_limiters:
            # Keep only the most recent half
            keep_count = max_limiters // 2
            self.user_limiters = dict(list(self.user_limiters.items())[-keep_count:])
        
        if len(self.ip_limiters) > max_limiters:
            keep_count = max_limiters // 2
            self.ip_limiters = dict(list(self.ip_limiters.items())[-keep_count:])

# FastAPI dependency for rate limiting
class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting"""
        from fastapi import Response
        from fastapi.responses import JSONResponse
        
        # Extract identifiers
        user_id = request.headers.get("X-User-ID")
        api_key = request.headers.get("X-API-Key")
        identifier = user_id or api_key
        
        # Get IP address
        ip_address = request.client.host if request.client else None
        
        # Check rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            ip_address=ip_address
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "message": f"Please try again in {retry_after} seconds"
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.rate_limiter.global_limiter.tokens
        )
        
        return response

# Decorator for rate limiting functions
def rate_limit(
    requests_per_minute: int = 60,
    identifier_func: Optional[callable] = None
):
    """
    Decorator for rate limiting functions
    
    Args:
        requests_per_minute: Maximum requests per minute
        identifier_func: Function to extract identifier from arguments
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract identifier
            identifier = None
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            
            # Check rate limit
            allowed, retry_after = await limiter.check_rate_limit(identifier=identifier)
            
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after} seconds",
                    retry_after=retry_after
                )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Extract identifier
                identifier = None
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                
                # Check rate limit
                allowed, retry_after = loop.run_until_complete(
                    limiter.check_rate_limit(identifier=identifier)
                )
                
                if not allowed:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded. Retry after {retry_after} seconds",
                        retry_after=retry_after
                    )
                
                return func(*args, **kwargs)
            finally:
                loop.close()
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global rate limiter instance
global_rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_size=10
)
