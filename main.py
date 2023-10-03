import time
import json
from typing import List, Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import numpy as np


app = FastAPI()



def count_python(numbers):
    sign_changes = 0

    sign = 1 if numbers[0] >= 0 else -1

    for number in numbers:
        if (number >= 0 and sign < 0) or (number < 0 and sign > 0):
            sign_changes += 1
            sign = -sign

    return sign_changes


def count_numpy(numbers):
    signs = np.array([number < 0 for number in numbers])

    shifted_signs = np.roll(signs, -1)
    shifted_signs[-1] = shifted_signs[-2]

    xor_result = np.logical_xor(signs, shifted_signs)

    unique, counts = np.unique(xor_result, return_counts=True)

    count_dict = dict(zip(unique, counts))
    result = count_dict[True]

    return result


def count_chatgpt(numbers):
    # Convert the list to a numpy array and get the sign of each number
    arr = np.array(numbers)
    signs = np.sign(arr)

    # Count the number of sign changes
    sign_changes = np.abs(np.diff(signs)).sum() // 2

    return sign_changes




class Lead(BaseModel):
    name: str
    sample_count: Union[int, None] = Field(default=None)
    signal: List[int]


class ECG(BaseModel):
    ecg_id: int
    date: str
    user_fk: Union[str, None] = Field(default=None)
    leads: List[Lead]


DATABASE = {}

USERS_DATABASE = {
    "user": {
        "username": "user",
        "hashed_password": "fakehasheduser",
        "is_admin": False,
    },
    "admin": {
        "username": "admin",
        "hashed_password": "fakehashedadmin",
        "is_admin": True,
    },
}


def fake_hash_password(password: str):
    return "fakehashed" + password


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


def _create_user(new_user: UserInDB):
    USERS_DATABASE[new_user.username] = {
        "username": new_user.username,
        "hashed_password": new_user.hashed_password,
        "is_admin": False,
    }


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def fake_decode_token(token):
    user = get_user(USERS_DATABASE, token)
    return user


def zero_crosses(l: Lead):
    result = {
        'name': l.name,
        'zero_crosses': count_python(l.signal)
    }
    return result


INSIGHT_CREATORS = {
    'zero_crosses': zero_crosses
}


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    user = fake_decode_token(token)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_admin_user(current_user: Annotated[User, Depends(get_current_user)]):
    user = USERS_DATABASE[current_user.username]

    if not user['is_admin']:
        raise HTTPException(status_code=400, detail="Non admin users cannot add new users")

    return current_user


async def get_normal_user(current_user: Annotated[User, Depends(get_current_user)]):
    user = USERS_DATABASE[current_user.username]

    if user['is_admin']:
        raise HTTPException(status_code=400, detail="Admin users cannot interact with ECG")

    return current_user


@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user_dict = USERS_DATABASE.get(form_data.username)

    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)

    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}


@app.post("/users/create")
async def create_user(current_user: Annotated[User, Depends(get_admin_user)], new_user: UserInDB):
    _create_user(new_user)
    return {'success': True}


@app.post("/ecg/insert")
async def insert_ecg(current_user: Annotated[User, Depends(get_normal_user)], ecg: ECG):
    # print(f'creating ecg with id: {ecg.ecg_id}')

    for l in ecg.leads:
        if l.sample_count is None:
            l.sample_count = len(l.signal)

    ecg.user_fk = current_user.username
    DATABASE[ecg.ecg_id] = ecg
    # print(json.dumps(DATABASE, indent=2, default=str))
    # print(f'creation success')
    return {'success': True}


@app.get("/ecg/{id}/{insight_type}")
async def get_insights(current_user: Annotated[User, Depends(get_normal_user)], ecg_id: int, insight_type: str):
    user = USERS_DATABASE[current_user.username]

    ecg = DATABASE.get(ecg_id)

    if ecg.user_fk != current_user.username:
        raise HTTPException(status_code=400, detail="ECG does not belong to this user")

    # print(json.dumps(DATABASE, indent=2, default=str))

    if not ecg:
        print('insight not found')
        return {}

    insight_creator = INSIGHT_CREATORS[insight_type]
    result = [
        insight_creator(l)
        for l in ecg.leads
    ]

    return result


def generate_random_signed_integers(low, high, size):
    random_integers = np.random.randint(low, high, size)
    random_integers_list = random_integers.tolist()
    return random_integers_list


if __name__ == '__main__':
    # Testing a couple of methods to see which is faster
    numbers = generate_random_signed_integers(-100, 100, 30000000)

    start_time = time.time()
    print(count_numpy(numbers))
    end_time = time.time()
    print(f"count_numpy: {end_time - start_time}")

    start_time = time.time()
    print(count_chatgpt(numbers))
    end_time = time.time()
    print(f"count_chatgpt: {end_time - start_time}")

    start_time = time.time()
    print(count_python(numbers))
    end_time = time.time()
    print(f"count_python: {end_time - start_time}")




