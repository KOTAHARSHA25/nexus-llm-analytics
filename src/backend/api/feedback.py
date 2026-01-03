"""
Feedback API: Collects user ratings for analysis results.
This enables a feedback flywheel for continuous improvement.

Author: Nexus Analytics Team
Date: January 3, 2026
Purpose: Fix 10 - Enable user feedback collection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request model for submitting user feedback"""
    query: str = Field(..., description="The original query", example="What is the average sales?")
    result: str = Field(..., description="The analysis result shown to user")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    thumbs_up: Optional[bool] = Field(None, description="Quick thumbs up/down (true=up, false=down)")
    comment: Optional[str] = Field(None, description="Optional user comment/suggestion")
    filename: Optional[str] = Field(None, description="Data file that was analyzed")
    analysis_id: Optional[str] = Field(None, description="Analysis ID for tracking")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str
    feedback_id: str


class FeedbackStats(BaseModel):
    """Aggregate feedback statistics"""
    total: int
    avg_rating: Optional[float]
    thumbs_up_count: int
    thumbs_down_count: int
    thumbs_up_rate: Optional[float]
    recent_comments: list


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for an analysis result.
    Stores (Query, Result, Rating) triplets for future fine-tuning.
    
    This creates a feedback flywheel:
    - Users rate results → 
    - System identifies weak queries → 
    - Prompts/models improved → 
    - Better results → Higher ratings
    """
    try:
        # Generate unique feedback ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.query) % 10000:04d}"
        
        # Prepare feedback entry
        entry = {
            "id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "query": request.query,
            "result": request.result[:1000],  # Truncate very long results
            "rating": request.rating,
            "thumbs_up": request.thumbs_up,
            "comment": request.comment,
            "filename": request.filename,
            "analysis_id": request.analysis_id
        }
        
        # Determine feedback storage location
        # Try project root data/feedback first
        project_root = Path(__file__).parent.parent.parent.parent
        feedback_dir = project_root / 'data' / 'feedback'
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = feedback_dir / 'user_feedback.jsonl'
        
        # Append feedback as JSONL (one JSON object per line)
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        
        # Log feedback for monitoring
        emoji = "👍" if request.thumbs_up else "👎" if request.thumbs_up is False else "⭐"
        logger.info(f"Feedback recorded: {feedback_id} | Rating: {request.rating}/5 {emoji} | Query: {request.query[:50]}...")
        
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback! It helps us improve.",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(limit: int = 10):
    """
    Get aggregate feedback statistics.
    Useful for monitoring system performance and user satisfaction.
    
    Args:
        limit: Number of recent comments to include (default 10)
    """
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        feedback_file = project_root / 'data' / 'feedback' / 'user_feedback.jsonl'
        
        if not feedback_file.exists():
            return FeedbackStats(
                total=0,
                avg_rating=None,
                thumbs_up_count=0,
                thumbs_down_count=0,
                thumbs_up_rate=None,
                recent_comments=[]
            )
        
        total = 0
        ratings = []
        thumbs_up_count = 0
        thumbs_down_count = 0
        recent_comments = []
        
        # Read all feedback entries
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        total += 1
                        
                        # Collect ratings
                        if 'rating' in entry:
                            ratings.append(entry['rating'])
                        
                        # Count thumbs
                        if entry.get('thumbs_up') is True:
                            thumbs_up_count += 1
                        elif entry.get('thumbs_up') is False:
                            thumbs_down_count += 1
                        
                        # Collect recent comments
                        if entry.get('comment'):
                            recent_comments.append({
                                'comment': entry['comment'],
                                'rating': entry.get('rating'),
                                'timestamp': entry.get('timestamp'),
                                'query': entry.get('query', '')[:50] + '...'
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed feedback line: {line[:100]}")
                        continue
        
        # Calculate statistics
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        thumbs_total = thumbs_up_count + thumbs_down_count
        thumbs_up_rate = thumbs_up_count / thumbs_total if thumbs_total > 0 else None
        
        # Return most recent comments first
        recent_comments = sorted(
            recent_comments,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:limit]
        
        return FeedbackStats(
            total=total,
            avg_rating=round(avg_rating, 2) if avg_rating else None,
            thumbs_up_count=thumbs_up_count,
            thumbs_down_count=thumbs_down_count,
            thumbs_up_rate=round(thumbs_up_rate, 2) if thumbs_up_rate else None,
            recent_comments=recent_comments
        )
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@router.get("/export")
async def export_feedback(format: str = "jsonl"):
    """
    Export all feedback for analysis/fine-tuning.
    
    Args:
        format: Export format (jsonl, json, csv)
    
    Returns:
        Feedback data in requested format
    """
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        feedback_file = project_root / 'data' / 'feedback' / 'user_feedback.jsonl'
        
        if not feedback_file.exists():
            return {"entries": [], "count": 0}
        
        entries = []
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if format == "json":
            return {"entries": entries, "count": len(entries)}
        elif format == "csv":
            # Simple CSV format
            import io
            import csv
            output = io.StringIO()
            if entries:
                writer = csv.DictWriter(output, fieldnames=entries[0].keys())
                writer.writeheader()
                writer.writerows(entries)
            return {"csv": output.getvalue(), "count": len(entries)}
        else:  # jsonl (default)
            return {
                "jsonl": "\n".join(json.dumps(e) for e in entries),
                "count": len(entries)
            }
        
    except Exception as e:
        logger.error(f"Failed to export feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.delete("/reset")
async def reset_feedback(confirm: str = ""):
    """
    Reset/clear all feedback data (admin function).
    Requires confirmation parameter.
    
    Args:
        confirm: Must be "CONFIRM_RESET" to actually delete
    """
    if confirm != "CONFIRM_RESET":
        raise HTTPException(
            status_code=400,
            detail="Reset requires confirmation. Add ?confirm=CONFIRM_RESET"
        )
    
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        feedback_file = project_root / 'data' / 'feedback' / 'user_feedback.jsonl'
        
        if feedback_file.exists():
            # Backup before deleting
            backup_file = feedback_file.with_suffix('.jsonl.backup')
            feedback_file.rename(backup_file)
            logger.warning(f"Feedback reset - backup saved to {backup_file}")
            
            return {
                "success": True,
                "message": "Feedback data reset. Backup saved.",
                "backup_location": str(backup_file)
            }
        else:
            return {"success": True, "message": "No feedback data to reset"}
        
    except Exception as e:
        logger.error(f"Failed to reset feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
