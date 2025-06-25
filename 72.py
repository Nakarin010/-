import re
import collections
import pandas as pd
import time
import threading
import random
import logging
from datetime import datetime, timedelta
from together import Together
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            self.requests = [t for t in self.requests if t > cutoff]
            if len(self.requests) >= self.max_requests:
                wait = (self.requests[0] + timedelta(minutes=1) - now).total_seconds()
                if wait > 0:
                    time.sleep(wait)
            self.requests.append(datetime.now())

rate_limiter = RateLimiter(60)

class AIAgents:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)

agents = AIAgents(api_key=keys)

PROMPT_SYS_BASE = (
    "You are a world-class financial analyst. Follow these rules exactly:\n"
    "1. Think step by step, label your reasoning section as 'Reasoning:'.\n"
    "2. At the end, output exactly one line beginning with 'FINAL ANSWER:' and one of [A, B, C, D, Rise, Fall].\n"
    "3. Do NOT include any additional text after the final answer line.\n"
    "4. Keep reasoning concise but thorough, using bullet points if helpful.\n"
)

FEW_SHOT = (
    "Example 1:\n"
    "Question: P/E jumps from 8 to 16; what happens to price?\n"
    "Reasoning:\n"
    "- Higher P/E indicates investors pay more per earnings unit.\n"
    "- Demand likely pushes price up.\n"
    "FINAL ANSWER: Rise\n\n"

    "Example 2:\n"
    "Question: Earnings drop 20%, share count flat; price reaction?\n"
    "Reasoning:\n"
    "- Lower earnings reduce valuation.\n"
    "- Price likely declines.\n"
    "FINAL ANSWER: Fall\n\n"

    "Example 3:\n"
    "Question: Debt-to-equity spikes above 2x; outlook?\n"
    "Reasoning:\n"
    "- Higher leverage increases risk.\n"
    "- Investors may sell, causing price drop.\n"
    "FINAL ANSWER: Fall\n\n"

    "Example 4:\n"
    "Question: ROE improves from 5% to 10%; stock trend?\n"
    "Reasoning:\n"
    "- Improved ROE signals better profitability.\n"
    "- Likely attracts buyers, raising price.\n"
    "FINAL ANSWER: Rise\n\n"

    "Example 5:\n"
    "Question: Dividend yield increases but revenue flat; next move?\n"
    "Reasoning:\n"
    "- Higher yield may attract income investors.\n    - Flat revenue not offsetting yield appeal.\n"
    "FINAL ANSWER: Rise"
)

class ChoiceExtractor:
    final_pattern = re.compile(r"(?:FINAL\s*ANSWER)\s*[:\-]\s*([ABCD]|Rise|Fall)", re.IGNORECASE)
    last_pattern = re.compile(r"\b([ABCD]|Rise|Fall)\b(?!.*\b(?:[ABCD]|Rise|Fall)\b)", re.IGNORECASE)
    conf_pattern = re.compile(r"confidence\s*[:=]\s*(high|medium|low)", re.IGNORECASE)

    def extract(self, text: str):
        t = text.strip()
        m = self.final_pattern.search(t)
        choice = m.group(1) if m else None
        if not choice:
            m2 = self.last_pattern.search(t)
            choice = m2.group(1) if m2 else None
        if choice:
            choice = choice.capitalize() if choice.lower() in ['rise','fall'] else choice.upper()
        conf = 0.5
        m3 = self.conf_pattern.search(t)
        if m3:
            conf_map = {'high':1.0,'medium':0.7,'low':0.4}
            conf = conf_map.get(m3.group(1).lower(),0.5)
        return choice, conf

extractor = ChoiceExtractor()

def select_temps(query):
    length = len(query)
    if length < 50:
        return [0.1, 0.3]
    elif length < 100:
        return [0.3, 0.5, 0.7]
    else:
        return [0.5, 0.7, 0.9]

def override_heuristic(result, query):
    q = query.lower()
    if 'rise' in q and result == 'Fall':
        return 'Rise'
    if 'fall' in q and result == 'Rise':
        return 'Fall'
    return result

class TyphoonPredictor:
    def __init__(self, client, limiter):
        self.client = client
        self.limiter = limiter

    def predict(self, full_prompt, query):
        temps = select_temps(query)
        def run_round(temps_list):
            votes = collections.defaultdict(float)
            for i, t in enumerate(temps_list):
                self.limiter.wait_if_needed()
                resp = self.client.chat.completions.create(
                    model="scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
                    messages=[{'role':'system','content':full_prompt}, {'role':'user','content':query}],
                    temperature=t, max_tokens=512
                )
                text = resp.choices[0].message.content
                if i == 0:
                    logging.info(f"RAW @ t={t}: {text}")
                choice, conf = extractor.extract(text)
                if choice:
                    votes[choice] += conf
            if not votes:
                return None, votes
            total = sum(votes.values())
            scores = {c: v/total for c,v in votes.items()}
            winner = max(scores, key=scores.get)
            return winner, scores

        full_prompt = PROMPT_SYS_BASE + FEW_SHOT
        result, scores = run_round(temps)
        # if low confidence, re-query at higher temp
        max_conf = max(scores.values()) if scores else 0
        if max_conf < 0.3:
            logging.info("Low confidence, re-querying with higher temps")
            high_temps = [min(1.0, t+0.2) for t in temps]
            result, scores = run_round(high_temps)
        if not result:
            return None
        result = override_heuristic(result, query)
        return result

LABELS = ['A','B','C','D','Rise','Fall']

def fallback(query):
    q = query.lower()
    if any(w in q for w in ['rise','increase','up']): return 'Rise'
    if any(w in q for w in ['fall','decline','down']): return 'Fall'
    return random.choice(LABELS)

def run(test_df):
    predictor = TyphoonPredictor(agents.client, rate_limiter)
    full_prompt = PROMPT_SYS_BASE + FEW_SHOT
    results = [None] * len(test_df)
    def task(i, q):
        res = predictor.predict(full_prompt, q)
        return i, res if res else fallback(q)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
        futures = [exe.submit(task, idx, row['query']) for idx, row in test_df.iterrows()]
        for f in concurrent.futures.as_completed(futures):
            i, ans = f.result()
            results[i] = ans 
    return results

if __name__ == '__main__':
    df_test = pd.read_csv('test.csv')
    preds = run(df_test)
    df_sub = pd.read_csv('submission.csv')
    df_sub['answer'] = preds
    df_sub.to_csv('my_submission_typhoon_improved_v2.csv', index=False)
    logging.info('Saved v2 submission')
