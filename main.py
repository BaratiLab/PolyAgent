import subprocess
import sys
import os
from mcp.server.fastmcp import FastMCP, Context
from extensions import Register
from server_extensions import register_mcp_extensions
import prompts  # Import the new prompts module

mcp = FastMCP("PolyAgent")

def main():
    print("🚀 Starting PolyAgent MCP Coordinator...")

    # Initialize the MCP server registration system
    register = Register()

    # Register all extensions (prompts, tools, resources)
    register_mcp_extensions(register)

    # Register the new decorator-based prompts
    prompts.register_prompts_with_system(register)

    print(f"✅ Registered {len(register.get_prompts())} prompts")
    print(f"✅ Registered {len(register.get_tools())} tools")
    print(f"✅ Registered {len(register.get_resources())} resources")

    # Start the MCP servers if not already running
    mcp_servers =  {
      "Generation_mcp": {
        "command": "/home/vani/.local/bin/uv",
        "args": [
          "--directory",
          "/home/vani/mcp_servers/polyAgent/OMG_mcp",
          "run",
        "mcp",
        "run",
        "main.py"
      ]
    },
    "Property_mcp": {
      "command": "/home/vani/.local/bin/uv",
      "args": [
        "--directory",
        "/home/vani/mcp_servers/polyAgent/TransPolymer_mcp",
        "run",
        "mcp",
        "run",
        "main.py"
      ]
    }
    }

    # Check if MCP servers are already running
    running_servers = []
    for server in mcp_servers:
        try:
            # Try to check if the server process is running
            # For now, we'll assume they need to be started
            print(f"🔄 Starting {server}...")
            subprocess.Popen(mcp_servers[server]["command"], cwd=os.path.dirname(os.path.abspath(__file__)))
            running_servers.append(server)
        except Exception as e:
            print(f"⚠️  Could not start {server}: {e}")

    print(f"🎯 PolyAgent ready with {len(running_servers)} MCP servers connected")
    print("📡 Waiting for property optimization requests...")

    # Keep the main process running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n🛑 Shutting down PolyAgent...")
        sys.exit(0)

if __name__ == "__main__":
    # Run the FastMCP server with stdio transport
    mcp.run(transport="stdio")
