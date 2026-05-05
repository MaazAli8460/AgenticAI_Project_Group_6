from mcp.server.fastmcp import FastMCP
from agents.edit_agent.executor import (
    update_scene_background,
    update_character_visual,
    update_voice_tone,
    update_scene_bgm,
    run_pipeline_phase,
    get_project_context,
    get_asset_path
)

# Initialize the FastMCP server
mcp = FastMCP("VideoPipelineEditor")

# Expose the same tools the LangGraph agent uses as standard MCP tools
# This allows external clients (like Claude Desktop) to connect and edit the project natively.
mcp.tool()(update_scene_background)
mcp.tool()(update_character_visual)
mcp.tool()(update_voice_tone)
mcp.tool()(update_scene_bgm)
mcp.tool()(run_pipeline_phase)
mcp.tool()(get_project_context)
mcp.tool()(get_asset_path)

if __name__ == "__main__":
    # Start the standard stdio MCP server loop
    mcp.run()
