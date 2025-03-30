import asyncio

from .manager import ResearchManager

async def main() -> None:
    query = input("Enter a research query: ")
    mgr = ResearchManager()
    await mgr.run(query)


if __name__ == "__main__":
    asyncio.run(main())
