from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class UserLogin(BaseModel):
    email: str
    password: str

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class Token(BaseModel):
    token: str
    user: dict

# Dummy user data for testing
USERS = {
    "test@example.com": {
        "id": "1",
        "name": "Test User",
        "email": "test@example.com",
        "password": "password123"  # In production, use hashed passwords!
    }
}

@router.post("/login")
async def login(user_data: UserLogin):
    if user_data.email not in USERS or USERS[user_data.email]["password"] != user_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = USERS[user_data.email].copy()
    del user["password"]
    
    return Token(
        token="dummy_token",  # In production, use JWT tokens
        user=user
    )

@router.post("/register")
async def register(user_data: UserCreate):
    if user_data.email in USERS:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "id": str(len(USERS) + 1),
        "name": user_data.name,
        "email": user_data.email,
        "password": user_data.password  # In production, hash the password
    }
    
    USERS[user_data.email] = new_user
    user_response = new_user.copy()
    del user_response["password"]
    
    return Token(
        token="dummy_token",  # In production, use JWT tokens
        user=user_response
    )