import asyncio
from prisma import Prisma
import os
from dotenv import load_dotenv

async def main():
    load_dotenv()
    db = Prisma()
    print(f"Connecting to database...")
    try:
        await db.connect()
        print("Successfully connected to database!")
        users = await db.profile.find_many(take=1)
        print(f"Fetched {len(users)} profiles.")
    except Exception as e:
        print(f"Error during connection: {type(e).__name__}: {e}")
    finally:
        if db.is_connected():
            await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
