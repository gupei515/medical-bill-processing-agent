from __future__ import annotations
import json, os, sys, time, traceback, logging
import glob, fitz 
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple


from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.schema import BaseRetriever

# Import the reasoning module
from reason_decide import create_reasoning_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

#build the vector store
db = FAISS(
    open("knowledge/icd10_knowledge.txt").read().splitlines(),
    embeddings,
)

#load pdf and get text
def load_next_pdf() -> str:
    pdfs = glob.glob("incoming_claims/*.pdf")
    if not pdfs:
        raise FileNotFoundError("No new claims found.")
    path = pdfs[0]                       # take the oldest/newest as you prefer
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    return text

schema_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a medical claims extractor. "
     "Return a JSON object with keys: claim_id, patient_name, dos, cpt, icd10, charges, policy_no. "
     "Make charges a float (USD)."),
    ("human", "{ocr_text}")
])

extract_chain = LLMChain(llm=llm, prompt=schema_prompt)

def ingest_agent() -> Dict[str, Any]:
    ocr = load_next_pdf()
    result_json = extract_chain.invoke({"ocr_text": ocr})
    return json.loads(result_json)

def basic_coverage_check(parsed: Dict[str, Any]) -> str:
    covered_cpts = {"99213", "99214"}
    if parsed["cpt"] in covered_cpts and float(parsed["charges"]) < 250:
        return "APPROVE"
    return "REVIEW"

def decision_tool(parsed: Dict[str, Any]) -> Dict[str, Any]:
    status = basic_coverage_check(parsed)
    if status === "APPROVE":
        payout =  parsed["charges"] * 0.8
        reason = "Found CPT in auto-approve table"
    else
        answer = qa.invoke({"question": f"Is {parsed['cpt']} covered for {parsed['patient_name']}?"})
        approve = re.search(r"\byes\b", answer["result"], re.I)；
        status = "APPROVE" if approve else "DENY"
        payout = parse["charges"] * (0.8 if approve else 0)
        reason = answer["result"]
    return {"status": status, "payout": round(payout, 2), "reason": reason}


