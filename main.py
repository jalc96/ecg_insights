import time
import json
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
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
    leads: List[Lead]


DATABASE = {}


def zero_crosses(l: Lead):
    result = {
        'name': l.name,
        'zero_crosses': count_python(l.signal)
    }
    return result


INSIGHT_CREATORS = {
    'zero_crosses': zero_crosses
}


@app.post("/ecg/insert")
async def create_ecg(ecg: ECG):
    print(f'creating ecg with id: {ecg.ecg_id}')

    for l in ecg.leads:
        if l.sample_count is None:
            l.sample_count = len(l.signal)

    DATABASE[ecg.ecg_id] = ecg
    print(f'creation success')


@app.get("/ecg/{id}/{insight_type}")
async def get_insights(ecg_id: int, insight_type: str):
    ecg = DATABASE.get(ecg_id)
    print(json.dumps(DATABASE, indent=2, default=str))

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
