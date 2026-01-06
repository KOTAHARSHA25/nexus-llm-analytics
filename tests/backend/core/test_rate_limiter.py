import pytest
import asyncio
import time
from unittest.mock import patch
from src.backend.core.rate_limiter import TokenBucket, RateLimiter, RateLimitStrategy, RateLimitExceeded, rate_limit

@pytest.mark.asyncio
async def test_token_bucket():
    bucket = TokenBucket(capacity=2, refill_rate=0.1, refill_amount=1) # Refills 10/sec
    
    # Consume 2
    ok, wait = await bucket.consume(2)
    assert ok is True
    
    # Next should fail
    ok, wait = await bucket.consume(1)
    assert ok is False
    assert wait > 0
    
    # Wait for refill (mock time)
    bucket.last_refill = time.time() - 0.2 # 2 refills
    ok, wait = await bucket.consume(1)
    assert ok is True

@pytest.mark.asyncio
async def test_rate_limiter_global():
    limiter = RateLimiter(requests_per_minute=60, burst_size=1)
    
    ok, _ = await limiter.check_rate_limit()
    assert ok is True
    
    ok, retry = await limiter.check_rate_limit()
    assert ok is False
    assert retry > 0

@pytest.mark.asyncio
async def test_rate_limiter_user():
    # Use burst_size=5 so global doesn't block before we test user-specific limits
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    
    # User A - first request, uses burst_size
    ok, _ = await limiter.check_rate_limit(identifier="userA")
    assert ok is True
    
    # User B should also be ok (separate user limit)
    ok, _ = await limiter.check_rate_limit(identifier="userB")
    assert ok is True
    
    # Now test user-specific exhaustion by hitting the same user multiple times
    # Each user gets their own limiter with the same burst_size (5)
    # So we need 5 more requests from userA to exhaust their bucket
    for _ in range(4):  # Already used 1
        await limiter.check_rate_limit(identifier="userA")
    
    # Now userA should be blocked
    ok, _ = await limiter.check_rate_limit(identifier="userA")
    assert ok is False

@pytest.mark.asyncio
async def test_decorator_async():
    @rate_limit(requests_per_minute=600) # Fast enough
    async def my_func():
        return "ok"
    
    assert await my_func() == "ok"

def test_decorator_sync():
    # Sync wrapper uses asyncio.run new loop. 
    # Be careful running this inside pytest async test which already has a loop?
    # No, this test function is sync, so it's fine.
    
    @rate_limit(requests_per_minute=600)
    def my_sync_func():
        return "ok"
    
    assert my_sync_func() == "ok"
