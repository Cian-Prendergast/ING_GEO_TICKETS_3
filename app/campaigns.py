"""
Campaigns Module - Ticket creation and management interface
Split from main.py to focus on campaign/ticket workflow
"""

from fasthtml.common import *
from monsterui.all import *
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Data directory setup (shared with main)
DATA_DIR = Path("data")
TICKETS_DIR = DATA_DIR / "tickets"

# Constants
class TicketStatus:
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class OptimizationChannel:
    ORGANIC = "organic"
    AI_OVERVIEW = "ai_overview"
    PEOPLE_ALSO_ASK = "people_also_ask"

# Available optimization channels
CHANNELS = [
    {"id": OptimizationChannel.ORGANIC, "name": "Organic Search", "description": "Improve organic search rankings"},
    {"id": OptimizationChannel.AI_OVERVIEW, "name": "AI Overview", "description": "Optimize for AI Overview inclusion"},
    {"id": OptimizationChannel.PEOPLE_ALSO_ASK, "name": "People Also Ask", "description": "Target PAA sections"}
]

def save_ticket(ticket_data: Dict) -> str:
    """Save ticket to data directory"""
    ticket_id = str(uuid.uuid4())
    ticket_data['id'] = ticket_id
    ticket_data['created_at'] = datetime.now().isoformat()
    ticket_data['status'] = TicketStatus.ACTIVE
    ticket_data['progress'] = 0
    
    # Ensure tickets directory exists
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    
    ticket_file = TICKETS_DIR / f"{ticket_id}.json"
    with open(ticket_file, 'w') as f:
        json.dump(ticket_data, f, indent=2)
    
    return ticket_id

def load_ticket(ticket_id: str) -> Optional[Dict]:
    """Load specific ticket"""
    ticket_file = TICKETS_DIR / f"{ticket_id}.json"
    if ticket_file.exists():
        try:
            with open(ticket_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ticket {ticket_id}: {e}")
    return None

def load_all_tickets():
    """Load all tickets from data directory - local copy for main.py"""
    tickets = []
    tickets_dir = DATA_DIR / "tickets"
    
    if tickets_dir.exists():
        for ticket_file in tickets_dir.glob("*.json"):
            try:
                with open(ticket_file, 'r') as f:
                    ticket = json.load(f)
                    ticket['id'] = ticket_file.stem
                    tickets.append(ticket)
                    print(f"✅ Loaded ticket: {ticket.get('name', 'Unnamed')} - Channels: {ticket.get('optimization_channels', [])}")
            except Exception as e:
                print(f"❌ Error loading ticket {ticket_file}: {e}")
    else:
        print(f"⚠️ Tickets directory not found: {tickets_dir}")
    
    print(f"📊 Total tickets loaded: {len(tickets)}")
    return sorted(tickets, key=lambda x: x.get('created_at', ''), reverse=True)


def check_agent_availability() -> Dict:
    """Check if trained agents are available"""
    judge_available = False
    optimizer_available = False
    
    # Check judge agent status
    judge_file = DATA_DIR / "judge_training" / "status.json"
    if judge_file.exists():
        try:
            with open(judge_file, 'r') as f:
                status = json.load(f)
                judge_available = status.get('status') == 'Production Ready'
        except:
            pass
    
    # Check optimizer agent status  
    optimizer_file = DATA_DIR / "optimization_iterations" / "status.json"
    if optimizer_file.exists():
        try:
            with open(optimizer_file, 'r') as f:
                status = json.load(f)
                optimizer_available = status.get('status') == 'Production Ready'
        except:
            pass
    
    return {
        'judge_available': judge_available,
        'optimizer_available': optimizer_available,
        'any_available': judge_available or optimizer_available
    }

def CreateTicketForm():
    """Ticket creation form"""
    agent_status = check_agent_availability()
    
    return Card(
        CardHeader(
            H2("Create New Optimization Ticket", cls=TextT.lg + TextT.bold)
        ),
        CardBody(
            Form(
                # Ticket Name
                Div(
                    LabelInput(
                        "Ticket Name",
                        id="ticket_name",
                        placeholder="e.g., 'Banking Services SEO Campaign'",
                        required=True,
                        cls="w-full"
                    ),
                    cls="mb-6"
                ),
                
                # Keywords Section
                Div(
                    Label("Target Keywords", cls=TextT.sm + TextT.bold + " block mb-2"),
                    Textarea(
                        placeholder="Enter keywords, one per line:\nbest banking services\ndigital banking solutions\nonline banking security",
                        id="keywords",
                        rows="4",
                        required=True,
                        cls="w-full p-3 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    ),
                    P("Enter one keyword per line. Start with 2-5 keywords for your first ticket.", 
                      cls=TextT.xs + TextT.muted),
                    cls="mb-6"
                ),
                
                # Optimization Channels
                Div(
                    Label("Optimization Channels", cls=TextT.sm + TextT.bold + " block mb-3"),
                    P("Select which channels you want to optimize for:", 
                      cls=TextT.xs + TextT.muted + " mb-3"),
                    
                    Div(
                        *[
                            Div(
                                Input(
                                    type="checkbox",
                                    name="channels",
                                    value=channel["id"],
                                    id=f"channel_{channel['id']}",
                                    cls="mr-3"
                                ),
                                Label(
                                    Div(
                                        Span(channel["name"], cls=TextT.sm + TextT.medium),
                                        P(channel["description"], cls=TextT.xs + TextT.muted),
                                    ),
                                    _for=f"channel_{channel['id']}",
                                    cls="cursor-pointer flex-1"
                                ),
                                cls="flex items-start p-3 border rounded hover:bg-gray-50 mb-2"
                            )
                            for channel in CHANNELS
                        ],
                        cls="space-y-1"
                    ),
                    cls="mb-6"
                ),
                
                # Agent Configuration
                Div(
                    Label("Agent Configuration", cls=TextT.sm + TextT.bold + " block mb-3"),
                    
                    # Show agent availability status
                    Div(
                        Alert(
                            Div(
                                Span("🤖", cls="mr-2"),
                                Span("Agent Status:", cls=TextT.sm + TextT.bold),
                                Ul(
                                    Li(f"Judge Agent: {'✅ Available' if agent_status['judge_available'] else '⏳ Not Ready'}"),
                                    Li(f"Optimizer Agent: {'✅ Available' if agent_status['optimizer_available'] else '⏳ Not Ready'}"),
                                    cls="mt-2 ml-4"
                                ),
                                cls="flex flex-col"
                            ),
                            type=AlertT.info if agent_status['any_available'] else AlertT.warning,
                            cls="mb-4"
                        )
                    ),
                    
                    # Agent selection (if available)
                    Div(
                        Label(
                            Input(
                                type="checkbox",
                                name="use_agents",
                                value="true",
                                checked=agent_status['any_available'],
                                disabled=not agent_status['any_available'],
                                cls="mr-2"
                            ),
                            Span("Use trained agents for optimization", cls=TextT.sm),
                            cls="flex items-center"
                        ),
                        P("Agents will automatically optimize content when available. Manual optimization available as fallback.",
                          cls=TextT.xs + TextT.muted + " mt-1"),
                        cls="mb-6"
                    ) if agent_status['any_available'] else Div(
                        P("🔧 Agents are not yet trained. Ticket will use manual optimization workflow.",
                          cls=TextT.sm + " p-3 bg-yellow-50 border border-yellow-200 rounded"),
                        cls="mb-6"
                    ),
                    cls="mb-6"
                ),
                
                # Submit Button
                Div(
                    Button(
                        "Create Ticket",
                        type="submit",
                        cls="px-6 py-3 bg-blue-500 text-white rounded hover:bg-blue-600 font-medium"
                    ),
                    A("Cancel", href="/", 
                      cls="ml-4 px-6 py-3 text-gray-600 hover:text-gray-800"),
                    cls="flex items-center"
                ),
                
                hx_post="/api/tickets/create",
                hx_target="#form-result",
                hx_include="[name='ticket_name'], [name='keywords'], [name='channels'], [name='use_agents']"
            ),
            
            # Form result area
            Div(id="form-result", cls="mt-4"),
            
            cls="max-w-2xl"
        )
    )

def TicketsList():
    """List existing tickets"""
    tickets = load_all_tickets()
    
    if not tickets:
        return Card(
            CardBody(
                Div(
                    Div("📝", cls="text-4xl mb-4 text-center"),
                    H3("No tickets yet", cls=TextT.lg + TextT.medium + " text-center"),
                    P("Create your first optimization ticket to get started", 
                      cls=TextT.sm + TextT.muted + " text-center"),
                    cls="py-8 text-center"
                )
            )
        )
    
    def status_badge(status):
        colors = {
            TicketStatus.ACTIVE: "bg-green-100 text-green-800",
            TicketStatus.PAUSED: "bg-yellow-100 text-yellow-800",
            TicketStatus.COMPLETED: "bg-blue-100 text-blue-800",
            TicketStatus.FAILED: "bg-red-100 text-red-800"
        }
        return Span(
            status.title(),
            cls=f"px-2 py-1 text-xs rounded {colors.get(status, 'bg-gray-100 text-gray-800')}"
        )
    
    def format_channels(channels):
        channel_names = {
            OptimizationChannel.ORGANIC: "Organic",
            OptimizationChannel.AI_OVERVIEW: "AIO", 
            OptimizationChannel.PEOPLE_ALSO_ASK: "PAA"
        }
        return [channel_names.get(ch, ch) for ch in channels]
    
    def format_date(date_str):
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y")
        except:
            return date_str
    
    ticket_cards = []
    for ticket in tickets[:10]:  # Show recent 10
        channels = format_channels(ticket.get('optimization_channels', []))
        
        card = Card(
            CardBody(
                Div(
                    Div(
                        H4(ticket.get('name', 'Unnamed Ticket'), cls=TextT.medium),
                        status_badge(ticket.get('status', 'unknown')),
                        cls="flex justify-between items-start mb-2"
                    ),
                    
                    P(f"Keywords: {', '.join(ticket.get('keywords', [])[:3])}{'...' if len(ticket.get('keywords', [])) > 3 else ''}", 
                      cls=TextT.sm + TextT.muted + " mb-2"),
                    
                    Div(
                        Span("Channels: ", cls=TextT.xs + TextT.muted),
                        *[Span(ch, cls="px-1 py-0.5 text-xs bg-gray-100 rounded mr-1") for ch in channels],
                        cls="mb-3"
                    ),
                    
                    Div(
                        Span(f"Created: {format_date(ticket.get('created_at', ''))}", 
                             cls=TextT.xs + TextT.muted),
                        A("View Details", href=f"/campaigns/{ticket.get('id')}", 
                          cls="text-blue-500 hover:text-blue-700 text-sm"),
                        cls="flex justify-between items-center"
                    )
                )
            ),
            cls="mb-4"
        )
        ticket_cards.append(card)
    
    return Div(
        H3("Recent Tickets", cls=TextT.lg + TextT.bold + " mb-4"),
        *ticket_cards,
        
        A("View All Tickets", href="/", 
          cls="text-blue-500 hover:text-blue-700 text-sm"),
        cls="mt-8"
    )

# Campaign route handlers (to be imported by main.py)
def campaigns_home():
    """Main campaigns page"""
    return Title("Campaign Management"), Div(
        # Header
        Div(
            Div(
                H1("Campaign Management", cls=TextT.xl + TextT.bold + " mb-2"),
                P("Create and manage SEO optimization tickets", cls=TextT.sm + TextT.muted),
                cls="flex-1"
            ),
            Div(
                A("📊 Dashboard", href="/", 
                  cls="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 mr-2"),
                A("🔧 Train Agents", href="/agents", 
                  cls="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"),
                cls="flex items-center"
            ),
            cls="flex justify-between items-start mb-8 border-b pb-4"
        ),
        
        # Main content - two column layout
        Div(
            # Left column - Create ticket form
            Div(
                CreateTicketForm(),
                cls="lg:col-span-2"
            ),
            
            # Right column - Recent tickets
            Div(
                TicketsList(),
                cls="lg:col-span-1"
            ),
            
            cls="grid grid-cols-1 lg:grid-cols-3 gap-8"
        ),
        
        cls="container mx-auto px-4 py-8"
    )

def ticket_detail(ticket_id: str):
    """Individual ticket detail/monitoring page"""
    ticket = load_ticket(ticket_id)
    
    if not ticket:
        return Title("Ticket Not Found"), Div(
            H1("Ticket Not Found", cls=TextT.lg + TextT.bold),
            P("The requested ticket could not be found.", cls=TextT.sm + TextT.muted),
            A("← Back to Campaigns", href="/campaigns", cls="text-blue-500 hover:text-blue-700"),
            cls="container mx-auto px-4 py-8"
        )
    
    def format_channels(channels):
        channel_names = {
            OptimizationChannel.ORGANIC: "Organic Search",
            OptimizationChannel.AI_OVERVIEW: "AI Overview",
            OptimizationChannel.PEOPLE_ALSO_ASK: "People Also Ask"
        }
        return [channel_names.get(ch, ch) for ch in channels]
    
    return Title(f"Ticket: {ticket.get('name', 'Unnamed')}"), Div(
        # Header with back link
        Div(
            A("← Back to Campaigns", href="/campaigns", cls="text-blue-500 hover:text-blue-700 text-sm mb-2 block"),
            H1(ticket.get('name', 'Unnamed Ticket'), cls=TextT.xl + TextT.bold + " mb-2"),
            P(f"Created: {ticket.get('created_at', '')}", cls=TextT.sm + TextT.muted),
            cls="mb-8 border-b pb-4"
        ),
        
        # Ticket details
        Div(
            # Left column - Ticket info
            Div(
                Card(
                    CardHeader(
                        H3("Ticket Details", cls=TextT.lg + TextT.bold)
                    ),
                    CardBody(
                        Div(
                            P(f"Status: {ticket.get('status', 'Unknown').title()}", cls=TextT.sm + " mb-2"),
                            P(f"Progress: {ticket.get('progress', 0)}%", cls=TextT.sm + " mb-2"),
                            
                            Div(
                                P("Target Keywords:", cls=TextT.sm + TextT.bold + " mb-1"),
                                Ul(
                                    *[Li(keyword, cls=TextT.sm) for keyword in ticket.get('keywords', [])],
                                    cls="ml-4"
                                ),
                                cls="mb-4"
                            ),
                            
                            Div(
                                P("Optimization Channels:", cls=TextT.sm + TextT.bold + " mb-1"),
                                Ul(
                                    *[Li(channel, cls=TextT.sm) for channel in format_channels(ticket.get('optimization_channels', []))],
                                    cls="ml-4"
                                ),
                                cls="mb-4"
                            )
                        )
                    )
                ),
                cls="lg:col-span-1"
            ),
            
            # Right column - Actions and status
            Div(
                Card(
                    CardHeader(
                        H3("Actions", cls=TextT.lg + TextT.bold)
                    ),
                    CardBody(
                        P("⚡ Optimization monitoring and control will be available once the ticket processing system is implemented.", 
                          cls=TextT.sm + TextT.muted + " p-4 bg-blue-50 border border-blue-200 rounded"),
                        
                        Div(
                            Button("Pause Ticket", cls="px-4 py-2 bg-yellow-500 text-white rounded mr-2"),
                            Button("Resume Ticket", cls="px-4 py-2 bg-green-500 text-white rounded mr-2"),
                            Button("Stop Ticket", cls="px-4 py-2 bg-red-500 text-white rounded"),
                            cls="mt-4"
                        )
                    )
                ),
                cls="lg:col-span-1"
            ),
            
            cls="grid grid-cols-1 lg:grid-cols-2 gap-8"
        ),
        
        cls="container mx-auto px-4 py-8"
    )

async def create_ticket_handler(request):
    """Create new ticket API endpoint"""
    try:
        form = await request.form()
        
        # Extract form data
        ticket_name = form.get('ticket_name', '').strip()
        keywords_text = form.get('keywords', '').strip()
        selected_channels = form.getlist('channels')
        use_agents = form.get('use_agents') == 'true'
        
        # Validation
        if not ticket_name:
            return Alert("Please provide a ticket name", type=AlertT.error)
        
        if not keywords_text:
            return Alert("Please provide at least one keyword", type=AlertT.error)
        
        if not selected_channels:
            return Alert("Please select at least one optimization channel", type=AlertT.error)
        
        # Parse keywords
        keywords = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
        
        # Create ticket data
        ticket_data = {
            'name': ticket_name,
            'keywords': keywords,
            'optimization_channels': selected_channels,
            'use_agents': use_agents,
            'description': f"Optimizing {len(keywords)} keywords across {len(selected_channels)} channels"
        }
        
        # Save ticket
        ticket_id = save_ticket(ticket_data)
        
        return Div(
            Alert(
                Div(
                    H4("✅ Ticket Created Successfully!", cls=TextT.medium + " mb-2"),
                    P(f"Ticket ID: {ticket_id}", cls=TextT.sm + TextT.muted),
                    A("View Ticket", href=f"/campaigns/{ticket_id}", 
                      cls="text-blue-500 hover:text-blue-700 text-sm"),
                    A("View Dashboard", href="/", 
                      cls="ml-4 text-blue-500 hover:text-blue-700 text-sm")
                ),
                type=AlertT.success
            ),
            
            # Reset form with delay
            Script("""
                setTimeout(() => {
                    document.getElementById('ticket_name').value = '';
                    document.getElementById('keywords').value = '';
                    document.querySelectorAll('input[name="channels"]').forEach(cb => cb.checked = false);
                    document.querySelector('input[name="use_agents"]').checked = false;
                }, 2000);
            """)
        )
        
    except Exception as e:
        return Alert(f"Error creating ticket: {str(e)}", type=AlertT.error)