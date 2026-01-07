from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "SceneForge API"
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./test.db"
    frontend_url: str = "http://localhost:3000"  # Next.js default port
    celery_broker_url: str = "redis://default:2C5hxDYDL3Td7hg0kADZlbOzG0C7FIZW@redis-16058.c62.us-east-1-4.ec2.redns.redis-cloud.com:16058/0"
    celery_result_backend: str = "redis://default:2C5hxDYDL3Td7hg0kADZlbOzG0C7FIZW@redis-16058.c62.us-east-1-4.ec2.redns.redis-cloud.com:16058/0"

    class Config:
        env_file = ".env"

settings = Settings()