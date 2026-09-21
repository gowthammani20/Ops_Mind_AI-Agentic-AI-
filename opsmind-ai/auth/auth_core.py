"""
OpsMind AI — Authentication Core
OAuth2 + JWT + RBAC implementation
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import os

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# ─────────────────────────────────────────
# 1. RBAC Role Definitions
# ─────────────────────────────────────────

class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN       = "admin"
    ANALYST     = "analyst"
    VIEWER      = "viewer"
    AGENT       = "agent"          # Machine-to-machine (M2M) service account

# Permission matrix — each role maps to allowed actions
ROLE_PERMISSIONS: Dict[Role, List[str]] = {
    Role.SUPER_ADMIN: ["*"],                                # All permissions
    Role.ADMIN: [
        "read:metrics", "write:metrics",
        "read:anomalies", "write:anomalies",
        "read:predictions", "write:predictions",
        "manage:users", "manage:agents",
        "read:audit_logs",
    ],
    Role.ANALYST: [
        "read:metrics", "read:anomalies",
        "read:predictions", "write:predictions",
        "read:audit_logs",
    ],
    Role.VIEWER: [
        "read:metrics", "read:anomalies", "read:predictions",
    ],
    Role.AGENT: [
        "read:metrics", "write:metrics",
        "read:anomalies", "write:anomalies",
        "read:predictions",
    ],
}

# ─────────────────────────────────────────
# 2. Pydantic Schemas
# ─────────────────────────────────────────

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[Role] = None
    scopes: List[str] = []

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str

class UserInDB(BaseModel):
    username: str
    email: str
    full_name: str
    role: Role
    hashed_password: str
    is_active: bool = True
    created_at: datetime = datetime.utcnow()

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    role: Role = Role.VIEWER

class UserResponse(BaseModel):
    username: str
    email: str
    full_name: str
    role: Role
    is_active: bool

# ─────────────────────────────────────────
# 3. Password Hashing
# ─────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ─────────────────────────────────────────
# 4. In-Memory User Store (swap with DB in prod)
# ─────────────────────────────────────────

USERS_DB: Dict[str, UserInDB] = {
    "admin": UserInDB(
        username="admin",
        email="admin@opsmind.ai",
        full_name="System Administrator",
        role=Role.ADMIN,
        hashed_password=get_password_hash("admin123"),
    ),
    "analyst": UserInDB(
        username="analyst",
        email="analyst@opsmind.ai",
        full_name="Data Analyst",
        role=Role.ANALYST,
        hashed_password=get_password_hash("analyst123"),
    ),
    "viewer": UserInDB(
        username="viewer",
        email="viewer@opsmind.ai",
        full_name="Dashboard Viewer",
        role=Role.VIEWER,
        hashed_password=get_password_hash("viewer123"),
    ),
    "gowtham": UserInDB(
        username="gowtham",
        email="gowtham@opsmind.ai",
        full_name="Gowtham — AI Engineer",
        role=Role.SUPER_ADMIN,
        hashed_password=get_password_hash("gowtham2026"),
    ),
}

# ─────────────────────────────────────────
# 5. JWT Token Functions
# ─────────────────────────────────────────

SECRET_KEY   = os.getenv("SECRET_KEY", "opsmind-dev-secret-key-32-chars-min!!")
ALGORITHM    = os.getenv("ALGORITHM", "HS256")
ACCESS_EXPIRE  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_EXPIRE = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_EXPIRE))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_EXPIRE)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role_str: str = payload.get("role", Role.VIEWER)
        if username is None:
            raise credentials_exception
        role = Role(role_str)
        perms = ROLE_PERMISSIONS.get(role, [])
        return TokenData(username=username, role=role, scopes=perms)
    except JWTError:
        raise credentials_exception

# ─────────────────────────────────────────
# 6. FastAPI Dependency Injection
# ─────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    token_data = decode_token(token)
    user = USERS_DB.get(token_data.username)
    if user is None or not user.is_active:
        raise HTTPException(status_code=404, detail="User not found or inactive")
    return user

def require_permission(permission: str):
    """Dependency factory — enforces RBAC permission check."""
    async def _check(current_user: UserInDB = Depends(get_current_user)):
        perms = ROLE_PERMISSIONS.get(current_user.role, [])
        if "*" in perms or permission in perms:
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: requires '{permission}'",
        )
    return _check

def require_role(allowed_roles: List[Role]):
    """Dependency factory — enforces role-level access."""
    async def _check(current_user: UserInDB = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' not authorized for this resource",
            )
        return current_user
    return _check

# ─────────────────────────────────────────
# 7. OAuth2 Token Issue Endpoint Logic
# ─────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def issue_tokens(user: UserInDB) -> Token:
    payload = {"sub": user.username, "role": user.role.value}
    access_token  = create_access_token(payload)
    refresh_token = create_refresh_token(payload)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_EXPIRE * 60,
        role=user.role.value,
    )
