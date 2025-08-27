# Copyright 2025 Ant Group Inc.
import json
from typing import List, Tuple

from realhf.base import logging

import aiohttp

logger = logging.getLogger("Code ToolBox")

class CodeToolBox:

    async def step(self, qid_actions: Tuple[str, List[str]]):
        qid, actions = qid_actions

        results = []
        for action in actions:
            result = {}

            # tool calling
            if "<python>" in action and "</python>" in action:
                code = action.split("<python>")[-1].split("</python>")[0].strip()

                url = 'http://0.0.0.0:8080/run_code' # TODO
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                    "code": code,
                    "language": "python"
                }, timeout=90) as response:
                        response = await response.json()
                
                result["type"] = "python"
                result["response"] = response

            results.append(result)
        return results
