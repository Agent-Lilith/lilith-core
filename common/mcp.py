import logging

from mcp.server.fastmcp import FastMCP


def create_mcp_app(name: str) -> FastMCP:
    """Creates a FastMCP application instance with standardized logging."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    return FastMCP(name)


def run_mcp_server(mcp: FastMCP, transport: str = "stdio", port: int = 8000):
    """Runs the MCP server with the specified transport and port."""
    if transport == "stdio":
        mcp.run()
    else:
        # For HTTP transport
        mcp.run(transport="http", port=port)
