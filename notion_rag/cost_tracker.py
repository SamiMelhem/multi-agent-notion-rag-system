"""
Cost tracking for the Notion RAG system.
Monitors API usage costs for Gemini and other services.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Represents a single cost entry."""
    
    timestamp: float
    service: str  # "gemini", "notion", etc.
    operation: str  # "chat_completion", "embedding", etc.
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "service": self.service,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata
        }


class CostTracker:
    """Tracks API usage costs across different services."""
    
    # Gemini 2.5 Flash-Lite Preview pricing (as of 2024)
    GEMINI_PRICING = {
        "input_tokens_per_1m": 0.10,   # $0.10 per 1M input tokens
        "output_tokens_per_1m": 0.40,  # $0.40 per 1M output tokens
    }
    
    # Notion API pricing (if applicable)
    NOTION_PRICING = {
        "requests_per_1k": 0.008,  # $0.008 per 1K requests (if applicable)
    }
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the cost tracker.
        
        Args:
            log_file: Optional path to cost log file
        """
        self.log_file = log_file or "cost_log.json"
        self.entries: List[CostEntry] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.monthly_costs: Dict[str, float] = defaultdict(float)
        
        # Load existing cost data
        self._load_cost_data()
        
        logger.info(f"Cost tracker initialized. Log file: {self.log_file}")
    
    def _load_cost_data(self):
        """Load existing cost data from log file."""
        try:
            if Path(self.log_file).exists():
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    entries_data = data.get('entries', [])
                    
                    # Convert entries, handling the datetime field
                    self.entries = []
                    for entry_data in entries_data:
                        # Remove datetime field if present (it's derived from timestamp)
                        if 'datetime' in entry_data:
                            del entry_data['datetime']
                        self.entries.append(CostEntry(**entry_data))
                    
                    # Recalculate daily and monthly costs
                    self._recalculate_period_costs()
                    
                logger.info(f"Loaded {len(self.entries)} cost entries from {self.log_file}")
        except Exception as e:
            logger.warning(f"Failed to load cost data: {e}")
            # If loading fails, start with empty entries
            self.entries = []
    
    def _save_cost_data(self):
        """Save cost data to log file."""
        try:
            # Generate summary
            summary = self._generate_summary()
            
            data = {
                "entries": [entry.to_dict() for entry in self.entries],
                "summary": summary,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a detailed cost summary."""
        total_input_tokens = sum(entry.input_tokens for entry in self.entries)
        total_output_tokens = sum(entry.output_tokens for entry in self.entries)
        total_cost = sum(entry.cost_usd for entry in self.entries)
        
        # Calculate input and output costs separately
        input_cost = (total_input_tokens / 1_000_000) * self.GEMINI_PRICING["input_tokens_per_1m"]
        output_cost = (total_output_tokens / 1_000_000) * self.GEMINI_PRICING["output_tokens_per_1m"]
        
        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_entries": len(self.entries),
            "services": self.get_service_breakdown(),
            "operations": self.get_operation_breakdown()
        }
    
    def _recalculate_period_costs(self):
        """Recalculate daily and monthly costs from entries."""
        self.daily_costs.clear()
        self.monthly_costs.clear()
        
        for entry in self.entries:
            dt = datetime.fromtimestamp(entry.timestamp)
            day_key = dt.strftime("%Y-%m-%d")
            month_key = dt.strftime("%Y-%m")
            
            self.daily_costs[day_key] += entry.cost_usd
            self.monthly_costs[month_key] += entry.cost_usd
    
    def track_gemini_usage(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Track Gemini API usage and calculate cost.
        
        Args:
            operation: Operation type (e.g., "chat_completion", "rag_completion")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata
            
        Returns:
            float: Cost in USD
        """
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * self.GEMINI_PRICING["input_tokens_per_1m"]
        output_cost = (output_tokens / 1_000_000) * self.GEMINI_PRICING["output_tokens_per_1m"]
        total_cost = input_cost + output_cost
        
        # Create cost entry
        entry = CostEntry(
            timestamp=time.time(),
            service="gemini",
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
            metadata=metadata or {}
        )
        
        # Add to tracking
        self.entries.append(entry)
        
        # Update period costs
        dt = datetime.fromtimestamp(entry.timestamp)
        day_key = dt.strftime("%Y-%m-%d")
        month_key = dt.strftime("%Y-%m")
        
        self.daily_costs[day_key] += total_cost
        self.monthly_costs[month_key] += total_cost
        
        # Save to file
        self._save_cost_data()
        
        logger.info(f"Tracked Gemini usage: {operation}, "
                   f"Input: {input_tokens}, Output: {output_tokens}, "
                   f"Cost: ${total_cost:.6f}")
        
        return total_cost
    
    def track_notion_usage(
        self,
        operation: str,
        requests: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Track Notion API usage and calculate cost.
        
        Args:
            operation: Operation type (e.g., "fetch_pages", "fetch_blocks")
            requests: Number of API requests
            metadata: Additional metadata
            
        Returns:
            float: Cost in USD
        """
        # Calculate cost (if applicable)
        cost = (requests / 1000) * self.NOTION_PRICING["requests_per_1k"]
        
        # Create cost entry
        entry = CostEntry(
            timestamp=time.time(),
            service="notion",
            operation=operation,
            input_tokens=0,  # Notion doesn't use tokens
            output_tokens=0,
            cost_usd=cost,
            metadata=metadata or {}
        )
        
        # Add to tracking
        self.entries.append(entry)
        
        # Update period costs
        dt = datetime.fromtimestamp(entry.timestamp)
        day_key = dt.strftime("%Y-%m-%d")
        month_key = dt.strftime("%Y-%m")
        
        self.daily_costs[day_key] += cost
        self.monthly_costs[month_key] += cost
        
        # Save to file
        self._save_cost_data()
        
        logger.info(f"Tracked Notion usage: {operation}, "
                   f"Requests: {requests}, Cost: ${cost:.6f}")
        
        return cost
    
    def get_total_cost(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> float:
        """
        Get total cost for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            float: Total cost in USD
        """
        if not start_date and not end_date:
            return sum(entry.cost_usd for entry in self.entries)
        
        total = 0.0
        for entry in self.entries:
            entry_date = datetime.fromtimestamp(entry.timestamp).date()
            
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
                
            total += entry.cost_usd
        
        return total
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """
        Get daily costs for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict[str, float]: Daily costs
        """
        cutoff_date = date.today() - timedelta(days=days)
        return {day: cost for day, cost in self.daily_costs.items() 
                if datetime.strptime(day, "%Y-%m-%d").date() >= cutoff_date}
    
    def get_monthly_costs(self, months: int = 12) -> Dict[str, float]:
        """
        Get monthly costs for the last N months.
        
        Args:
            months: Number of months to look back
            
        Returns:
            Dict[str, float]: Monthly costs
        """
        cutoff_date = date.today() - timedelta(days=months * 30)
        return {month: cost for month, cost in self.monthly_costs.items() 
                if datetime.strptime(month, "%Y-%m").date() >= cutoff_date}
    
    def get_service_breakdown(self) -> Dict[str, float]:
        """
        Get cost breakdown by service.
        
        Returns:
            Dict[str, float]: Costs by service
        """
        breakdown = defaultdict(float)
        for entry in self.entries:
            breakdown[entry.service] += entry.cost_usd
        return dict(breakdown)
    
    def get_operation_breakdown(self) -> Dict[str, float]:
        """
        Get cost breakdown by operation.
        
        Returns:
            Dict[str, float]: Costs by operation
        """
        breakdown = defaultdict(float)
        for entry in self.entries:
            breakdown[entry.operation] += entry.cost_usd
        return dict(breakdown)
    
    def print_cost_summary(self):
        """Print a summary of current costs."""
        summary = self._generate_summary()
        
        print("\n💰 Cost Summary")
        print("=" * 50)
        print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
        print(f"Total Entries: {summary['total_entries']}")
        
        print(f"\n📊 Token Breakdown:")
        print(f"  Input Tokens: {summary['input_tokens']:,}")
        print(f"  Output Tokens: {summary['output_tokens']:,}")
        print(f"  Input Cost: ${summary['input_cost_usd']:.6f}")
        print(f"  Output Cost: ${summary['output_cost_usd']:.6f}")
        print(f"  Total Cost: ${summary['total_cost_usd']:.6f}")
        
        if summary['services']:
            print("\n🔧 By Service:")
            for service, cost in summary['services'].items():
                print(f"  {service.title()}: ${cost:.6f}")
        
        if summary['operations']:
            print("\n⚙️ By Operation:")
            for operation, cost in summary['operations'].items():
                print(f"  {operation}: ${cost:.6f}")
        
        # Show recent daily costs
        recent_daily = self.get_daily_costs(7)
        if recent_daily:
            print("\n📅 Last 7 Days:")
            for day, cost in sorted(recent_daily.items()):
                print(f"  {day}: ${cost:.6f}")
    
    def get_detailed_summary(self) -> Dict[str, Any]:
        """Get the detailed cost summary as a dictionary."""
        return self._generate_summary()


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def track_gemini_cost(operation: str, input_tokens: int, output_tokens: int, metadata: Optional[Dict[str, Any]] = None) -> float:
    """Convenience function to track Gemini costs."""
    return get_cost_tracker().track_gemini_usage(operation, input_tokens, output_tokens, metadata)


def track_notion_cost(operation: str, requests: int = 1, metadata: Optional[Dict[str, Any]] = None) -> float:
    """Convenience function to track Notion costs."""
    return get_cost_tracker().track_notion_usage(operation, requests, metadata) 