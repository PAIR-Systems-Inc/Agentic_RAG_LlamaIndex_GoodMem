"""Run the LlamaIndex documentation assistant and its acceptance cases."""

import argparse
import asyncio
import json
import sys

from .agents import ask
from .config import Settings, chat_model
from .ingestion import setup
from .retrieval import make_tools


async def run(args):
    settings = Settings.from_env()
    state = settings.state()
    async with settings.async_client() as client:
        tools = make_tools(
            async_client=client, state=state, rerank=not getattr(args, "no_rerank", False)
        )
        if args.command == "search":
            tool = next(t for t in tools if t.metadata.name == f"{args.collection}_docs_tool")
            print((await tool.acall(input=args.question)).content or "No results.")
        elif args.command == "ask":
            result = await ask(chat_model(), tools, args.question, agent=args.agent)
            print(result.rendered())
            print("\nSearches:", json.dumps(result.calls), file=sys.stderr)
        else:
            from .evaluation import evaluate

            if not await evaluate(settings, state, args.output, args.retrieval_only):
                raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    provision = commands.add_parser("setup", help="Load the six documentation pages into GoodMem")
    provision.add_argument("--init", action="store_true")
    provision.add_argument("--with-reranker", action="store_true")
    search = commands.add_parser("search", help="Search a documentation collection")
    search.add_argument("collection", choices=["langgraph", "langchain"])
    search.add_argument("question")
    search.add_argument("--no-rerank", action="store_true")
    query = commands.add_parser("ask", help="Ask the documentation assistant")
    query.add_argument("question")
    query.add_argument("--agent", choices=["workflow", "react"], default="react")
    evaluation = commands.add_parser("evaluate", help="Run retrieval and agent acceptance cases")
    evaluation.add_argument("--output", default=".runtime/evaluation.json")
    evaluation.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if args.command == "setup":
        state = setup(Settings.from_env(), initialize=args.init, with_reranker=args.with_reranker)
        print(f"Ready: {len(state['documents'])} documents in {len(state['spaces'])} spaces.")
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
