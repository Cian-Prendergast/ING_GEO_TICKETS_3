"""
Dashboard Module - Read-only monitoring interface for tickets
Split from main.py to focus on monitoring and system overview
"""

from fasthtml.common import *
from monsterui.all import *
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Data directory setup (shared with main)
DATA_DIR = Path("data")
TICKETS_DIR = DATA_DIR / "tickets"

class TicketStatus:
    ACTIVE = "active"
    PAUSED = "paused" 
    COMPLETED = "completed"
    FAILED = "failed"

class OptimizationChannel:
    ORGANIC = "organic"
    AI_OVERVIEW = "ai_overview" 
    PEOPLE_ALSO_ASK = "people_also_ask"

def load_tickets() -> List[Dict]:
    """Load all tickets from data directory"""
    tickets = []
    if TICKETS_DIR.exists():
        for ticket_file in TICKETS_DIR.glob("*.json"):
            try:
                with open(ticket_file, 'r') as f:
                    ticket = json.load(f)
                    ticket['id'] = ticket_file.stem
                    tickets.append(ticket)
            except Exception as e:
                print(f"Error loading ticket {ticket_file}: {e}")
    return sorted(tickets, key=lambda x: x.get('created_at', ''), reverse=True)

def load_system_metrics() -> Dict:
    """Load system-wide metrics"""
    tickets = load_tickets()
    
    # Calculate basic metrics
    total_tickets = len(tickets)
    active_tickets = len([t for t in tickets if t.get('status') == TicketStatus.ACTIVE])
    completed_tickets = len([t for t in tickets if t.get('status') == TicketStatus.COMPLETED])
    
    # Calculate channel distribution
    channels = {}
    for ticket in tickets:
        for channel in ticket.get('optimization_channels', []):
            channels[channel] = channels.get(channel, 0) + 1
    
    # Calculate recent performance (mock data for now - will connect to real metrics)
    recent_performance = {
        'organic_improvements': 12,
        'aio_inclusions': 5,
        'paa_appearances': 8,
        'avg_improvement': 23.5
    }
    
    return {
        'total_tickets': total_tickets,
        'active_tickets': active_tickets,
        'completed_tickets': completed_tickets,
        'channels': channels,
        'performance': recent_performance,
        'last_updated': datetime.now().isoformat()
    }

def get_agent_status() -> Dict:
    """Get current agent training/availability status"""
    judge_status = "Not Trained"
    optimizer_status = "Not Trained" 
    
    # Check for agent status files (connecting to existing data structure)
    judge_file = DATA_DIR / "judge_training" / "status.json"
    if judge_file.exists():
        try:
            with open(judge_file, 'r') as f:
                status = json.load(f)
                judge_status = status.get('status', 'Unknown')
        except:
            pass
    
    # Optimizer availability depends on Judge Agent being ready
    if judge_status == "Production Ready":
        optimizer_status = "Available"
    else:
        # Also check for explicit optimizer status file
        optimizer_file = DATA_DIR / "optimization_iterations" / "status.json"
        if optimizer_file.exists():
            try:
                with open(optimizer_file, 'r') as f:
                    status = json.load(f)
                    optimizer_status = status.get('status', 'Unknown')
            except:
                pass
    
    return {
        'judge_agent': judge_status,
        'optimizer_agent': optimizer_status,
        'system_health': 'Operational',
        'last_check': datetime.now().isoformat()
    }

def SystemMetricsCards():
    """System overview metrics cards"""
    metrics = load_system_metrics()
    
    metric_cards = [
        {
            'title': 'Total Tickets',
            'value': str(metrics['total_tickets']),
            'subtitle': f"{metrics['active_tickets']} active",
            'color': 'blue'
        },
        {
            'title': 'Completed',
            'value': str(metrics['completed_tickets']),
            'subtitle': 'This month',
            'color': 'green'
        },
        {
            'title': 'Avg Improvement',
            'value': f"{metrics['performance']['avg_improvement']}%",
            'subtitle': 'Across all channels',
            'color': 'purple'
        },
        {
            'title': 'AIO Inclusions',
            'value': str(metrics['performance']['aio_inclusions']),
            'subtitle': 'Recent wins',
            'color': 'orange'
        }
    ]
    
    return Div(
        H2("System Overview", cls=TextT.lg + TextT.bold),
        Div(
            *[
                Card(
                    CardBody(
                        H3(card['value'], cls=TextT.xl + TextT.bold),
                        P(card['title'], cls=TextT.sm + TextT.muted),
                        P(card['subtitle'], cls=TextT.xs + TextT.muted)
                    ),
                    cls=f"border-l-4 border-{card['color']}-500"
                )
                for card in metric_cards
            ],
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
        )
    )

def AgentStatusPanel():
    """Agent availability and health status"""
    agent_status = get_agent_status()
    
    def status_badge(status):
        if status == "Production Ready":
            return Span("Ready", cls="px-2 py-1 text-xs bg-green-100 text-green-800 rounded")
        elif "Training" in status:
            return Span("Training", cls="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded")
        else:
            return Span("Not Ready", cls="px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded")
    
    return Card(
        CardHeader(
            Div(
                H3("Agent Status", cls=TextT.lg + TextT.bold),
                A("🔧 Train Agents", href="/agents", 
                  cls="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"),
                cls="flex justify-between items-center"
            )
        ),
        CardBody(
            Div(
                Div(
                    Span("🤖 Judge Agent", cls=TextT.sm + TextT.medium),
                    status_badge(agent_status['judge_agent']),
                    cls="flex justify-between items-center p-3 border rounded mb-2"
                ),
                Div(
                    Span("🎯 Optimizer Agent", cls=TextT.sm + TextT.medium),
                    status_badge(agent_status['optimizer_agent']),
                    cls="flex justify-between items-center p-3 border rounded mb-2"
                ),
                Div(
                    Span("⚡ System Health", cls=TextT.sm + TextT.medium),
                    Span("Operational", cls="px-2 py-1 text-xs bg-green-100 text-green-800 rounded"),
                    cls="flex justify-between items-center p-3 border rounded"
                ),
                
                # Training status and quick action
                Div(
                    P("Need to train agents?", cls=TextT.xs + TextT.muted + " mb-2"),
                    A("Start Agent Training →", href="/agents", 
                      cls="text-blue-500 hover:text-blue-700 text-sm font-medium"),
                    cls="mt-3 p-3 bg-blue-50 rounded border border-blue-200"
                ) if agent_status['judge_agent'] == 'Not Trained' else None
            )
        ),
        cls="mb-8"
    )

def TicketsTable():
    """Main tickets overview table"""
    tickets = load_tickets()
    
    if not tickets:
        return Card(
            CardHeader(
                Div(
                    H3("Active Tickets", cls=TextT.lg + TextT.bold),
                    A("Create New Ticket", href="/campaigns", 
                      cls="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"),
                    cls="flex justify-between items-center"
                )
            ),
            CardBody(
                Div(
                    Div("📝", cls="text-4xl mb-4 text-center"),
                    H4("No tickets found", cls=TextT.lg + TextT.medium + " text-center"),
                    P("Create your first optimization ticket to get started", 
                      cls=TextT.sm + TextT.muted + " text-center"),
                    cls="py-12 text-center"
                )
            )
        )
    
    def status_indicator(status):
        colors = {
            TicketStatus.ACTIVE: "bg-green-500",
            TicketStatus.PAUSED: "bg-yellow-500", 
            TicketStatus.COMPLETED: "bg-blue-500",
            TicketStatus.FAILED: "bg-red-500"
        }
        return Span(
            cls=f"w-3 h-3 rounded-full {colors.get(status, 'bg-gray-500')}"
        )
    
    def format_channels(channels):
        channel_names = {
            OptimizationChannel.ORGANIC: "Organic",
            OptimizationChannel.AI_OVERVIEW: "AIO",
            OptimizationChannel.PEOPLE_ALSO_ASK: "PAA"
        }
        return ", ".join([channel_names.get(ch, ch) for ch in channels])
    
    def format_date(date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str
    
    table_rows = []
    for ticket in tickets:
        row = Tr(
            Td(
                Div(
                    status_indicator(ticket.get('status', 'unknown')),
                    Span(ticket.get('name', 'Unnamed'), cls=TextT.sm + TextT.medium + " ml-2"),
                    cls="flex items-center"
                )
            ),
            Td(
                ", ".join(ticket.get('keywords', [])) if ticket.get('keywords') else "No keywords",
                cls=TextT.sm
            ),
            Td(
                format_channels(ticket.get('optimization_channels', [])),
                cls=TextT.xs
            ),
            Td(
                format_date(ticket.get('created_at', '')),
                cls=TextT.xs + TextT.muted
            ),
            Td(
                f"{ticket.get('progress', 0)}%",
                cls=TextT.sm
            ),
            Td(
                A("View", href=f"/campaigns/{ticket.get('id')}", 
                  cls="text-blue-500 hover:text-blue-700 text-sm")
            )
        )
        table_rows.append(row)
    
    return Card(
        CardHeader(
            Div(
                H3("Active Tickets", cls=TextT.lg + TextT.bold),
                A("Create New Ticket", href="/campaigns",
                  cls="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"),
                cls="flex justify-between items-center"
            )
        ),
        CardBody(
            Table(
                Thead(
                    Tr(
                        Th("Ticket", cls=TextT.xs + TextT.bold + " p-2"),
                        Th("Keywords", cls=TextT.xs + TextT.bold + " p-2"),
                        Th("Channels", cls=TextT.xs + TextT.bold + " p-2"),
                        Th("Created", cls=TextT.xs + TextT.bold + " p-2"),
                        Th("Progress", cls=TextT.xs + TextT.bold + " p-2"),
                        Th("Actions", cls=TextT.xs + TextT.bold + " p-2")
                    )
                ),
                Tbody(*table_rows),
                cls="w-full border-collapse"
            ),
            cls="overflow-x-auto"
        )
    )

def ChannelPerformanceChart():
    """Simple channel performance overview"""
    metrics = load_system_metrics()
    channels = metrics.get('channels', {})
    
    channel_display = {
        OptimizationChannel.ORGANIC: {"name": "Organic Search", "color": "bg-blue-500"},
        OptimizationChannel.AI_OVERVIEW: {"name": "AI Overview", "color": "bg-green-500"},
        OptimizationChannel.PEOPLE_ALSO_ASK: {"name": "People Also Ask", "color": "bg-purple-500"}
    }
    
    return Card(
        CardHeader(
            H3("Channel Distribution", cls=TextT.lg + TextT.bold)
        ),
        CardBody(
            Div(
                *[
                    Div(
                        Div(
                            Span(channel_display[channel]["name"], cls=TextT.sm),
                            Span(str(count), cls=TextT.sm + TextT.bold),
                            cls="flex justify-between mb-2"
                        ),
                        Div(
                            Div(cls=f"h-2 {channel_display[channel]['color']} rounded", 
                                style=f"width: {min(100, (count/max(channels.values(), default=1))*100)}%"),
                            cls="w-full bg-gray-200 rounded h-2"
                        ),
                        cls="mb-4"
                    )
                    for channel, count in channels.items()
                    if channel in channel_display
                ]
            )
        ),
        cls="mb-8"
    )

# Dashboard route handlers (to be imported by main.py)
def dashboard_home():
    """Main dashboard page"""
    return Title("SEO Optimization Dashboard"), Div(
        # Header
        Div(
            Div(
                H1("SEO Optimization Dashboard", cls=TextT.xl + TextT.bold + " mb-2"),
                P("Monitor your optimization tickets and system performance", 
                  cls=TextT.sm + TextT.muted),
                cls="flex-1"
            ),
            Div(
                A("🔧 Train Agents", href="/agents", 
                  cls="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mr-2"),
                A("📋 Create Ticket", href="/campaigns", 
                  cls="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"),
                cls="flex items-center"
            ),
            cls="flex justify-between items-start mb-8 border-b pb-4"
        ),
        
        # Main content
        SystemMetricsCards(),
        
        Div(
            Div(
                AgentStatusPanel(),
                ChannelPerformanceChart(),
                cls="lg:col-span-1"
            ),
            Div(
                TicketsTable(),
                cls="lg:col-span-2"
            ),
            cls="grid grid-cols-1 lg:grid-cols-3 gap-8"
        ),
        
        cls="container mx-auto px-4 py-8"
    )

def refresh_dashboard_data():
    """Refresh dashboard data (HTMX endpoint)"""
    return Div(
        SystemMetricsCards(),
        AgentStatusPanel(),
        TicketsTable(),
        id="dashboard-content"
    )
