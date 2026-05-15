from arxiv_rag.agent.prompts import SYSTEM_PROMPT


def test_system_prompt_mentions_retrieval_and_citations():
    assert "retrieval" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()


def test_build_agent_options_allows_only_search_tool():
    from arxiv_rag.agent.service import build_agent_options

    options = build_agent_options(mcp_server="server")

    assert options.allowed_tools == ["mcp__arxiv__search_arxiv_papers"]