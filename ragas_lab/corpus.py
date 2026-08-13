"""
corpus.py — the documents, and the questions we evaluate against.

Two corpora, kept apart, exactly as decided on Sprint 0 Day 3:

  POLICY  — the rules. Changes on a policy cycle. Every version retrievable.
  CASE    — this claim's documents. Fixed for a case.

Mixing them into one index is what lets a model quote a policy clause as if
it were a fact about the customer. Keep them separate and every citation
says which kind of source it came from.

THE EVALUATION SET
------------------
Twelve questions, each with:
    - the ground-truth answer
    - the chunk IDs that actually contain the answer (ground truth context)
    - a type, so you can see WHERE a configuration wins and loses

Types matter more than the average. A chunk size that is excellent at
single-fact lookup and useless at multi-part questions is not "worse" —
it is differently shaped, and knowing that is the point of the lab.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# POLICY CORPUS — clause-numbered, because clause numbers are how a human
# cites it and how a keyword search finds it.
# --------------------------------------------------------------------------
POLICY_DOCS: dict[str, str] = {
"POL-MOTOR-2026": """
CLAUSE 1.1 SCOPE
This policy covers accidental damage, fire and theft for the vehicle named in
the schedule. Cover applies within the United Kingdom and, for up to ninety
days in any period of insurance, within the European Economic Area.

CLAUSE 2.1 ACCIDENTAL DAMAGE LIMIT
The maximum amount payable for accidental damage in any one claim is the
market value of the vehicle at the time of loss, or the sum insured stated
in the schedule, whichever is lower.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule for each and every
claim. Where the driver at the time of the incident is under twenty-five
years of age, an additional young driver excess of GBP 250 applies.

CLAUSE 3.2 REPAIR AUTHORISATION
Repairs above GBP 5,000 require authorisation from a claims handler before
work begins. Repairs carried out without authorisation may be settled at the
amount the insurer would reasonably have paid an approved repairer.

CLAUSE 3.5 SUPPLEMENTARY ESTIMATES
Where additional damage is found after work has begun, a supplementary
estimate must be submitted. Supplementary work above twenty per cent of the
original estimate requires an engineer inspection before authorisation.

CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds sixty per cent of its pre-accident market value.

CLAUSE 5.3 UNINSURED DRIVER
No cover applies where the vehicle was being driven by a person not named on
the policy schedule and not covered by the driving other cars extension.

CLAUSE 6.2 CLAIM NOTIFICATION
A claim must be notified within thirty days of the incident. Late
notification does not automatically invalidate a claim, but the insurer may
reduce settlement by the amount of any prejudice caused by the delay.
""",

"POL-MOTOR-2025": """
CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule for each and every
claim. Where the driver at the time of the incident is under twenty-one years
of age, an additional young driver excess of GBP 150 applies.

CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds seventy per cent of its pre-accident market value.
""",
}

# --------------------------------------------------------------------------
# CASE CORPUS — one claim, several documents, deliberately messy.
# --------------------------------------------------------------------------
CASE_DOCS: dict[str, str] = {
"CLM-4471-notification": """
MOTOR CLAIM NOTIFICATION
Claim reference: CLM-2026-4471
Policy number: MOT-4471902
Policyholder: [PERSON_01]
Vehicle: 2019 Vauxhall Astra, registration KP19 TRX
Pre-accident market value: GBP 8,400.00
Incident date: 14 March 2026
Notified: 11 April 2026
Driver at time of incident: [PERSON_02], age 23, named on schedule
Description: Front nearside collision at low speed in a supermarket car park.
""",

"CLM-4471-schedule": """
POLICY SCHEDULE EXTRACT
Policy number: MOT-4471902
Sum insured: GBP 9,000.00
Standard excess: GBP 350.00
Named drivers: [PERSON_01], [PERSON_02]
Cover type: Comprehensive
Period of insurance: 1 January 2026 to 31 December 2026
""",

"CLM-4471-estimate": """
REPAIR ESTIMATE
Repairer: Northgate Motor Repairs Ltd
Date: 22 March 2026
  Front bumper assembly, replace ............ GBP 1,850.00
  Nearside wing, repair and paint ........... GBP 2,300.00
  Headlamp unit, replace .................... GBP 3,150.00
  Radiator support, straighten .............. GBP 1,900.00
  Labour, 22 hours @ GBP 145/hr ............. GBP 3,200.00
  TOTAL ..................................... GBP 12,400.00
Authorisation: not yet obtained at date of estimate.
""",

"CLM-4471-supplementary": """
SUPPLEMENTARY ESTIMATE
Repairer: Northgate Motor Repairs Ltd
Date: 4 April 2026
Additional structural damage found to the offside chassis leg on strip-down.
  Chassis leg section, replace .............. GBP 2,350.00
REVISED TOTAL ............................... GBP 14,750.00
""",

"CLM-4471-notes": """
HANDLER NOTES
23 Mar - Estimate received. Awaiting engineer availability.
06 Apr - Supplementary estimate received. Increase is 18.9% of original.
11 Apr - Claim formally notified on system. Note: incident was 14 March,
         so notification is 28 days after the incident.
14 Apr - Query raised: was repair work started before authorisation?
         Repairer confirms strip-down only, no repair work commenced.
""",
}


# --------------------------------------------------------------------------
# DISTRACTOR CLAIMS — the reason retrieval is hard.
#
# A real index does not hold one claim. It holds thousands, and they all look
# alike: same document types, same headings, same shaped numbers. This is the
# Session 2 problem in its natural habitat — the retriever has no idea which
# "REPAIR ESTIMATE ... TOTAL" belongs to the claim you asked about.
#
# Every distractor below is a well-formed claim. None is the one under review.
# --------------------------------------------------------------------------
_DISTRACTOR_SPEC = [
    ("4482", "LT21 WNB", "6,220.00", "9,100.00", "250.00", "2 April 2026", "Rear-end collision at a junction."),
    ("4490", "RV70 KDS", "9,800.00", "11,300.00", "400.00", "21 February 2026", "Offside impact with a stationary object."),
    ("4503", "BD18 MHK", "3,410.00", "7,750.00", "300.00", "9 March 2026", "Windscreen and nearside mirror damage."),
    ("4517", "GF22 PLT", "15,900.00", "12,000.00", "500.00", "28 March 2026", "Front and offside damage after a roundabout collision."),
    ("4526", "HN20 XCE", "2,150.00", "6,400.00", "250.00", "3 April 2026", "Rear bumper scuff in a multi-storey car park."),
    ("4534", "JW19 RQA", "11,050.00", "10,200.00", "350.00", "17 March 2026", "Nearside collision with a bollard."),
    ("4548", "MS21 TVB", "7,630.00", "8,900.00", "300.00", "25 March 2026", "Front nearside collision in a supermarket car park."),
    ("4552", "PC17 KDN", "4,880.00", "5,600.00", "250.00", "12 March 2026", "Wing mirror and door panel damage."),
    ("4561", "QT19 LFS", "13,200.00", "9,800.00", "400.00", "30 March 2026", "Offside front collision at a junction."),
    ("4573", "ZB20 HRC", "5,940.00", "7,100.00", "350.00", "6 April 2026", "Rear quarter panel damage, parked vehicle."),
]

for _ref, _reg, _total, _mv, _exc, _date, _desc in _DISTRACTOR_SPEC:
    CASE_DOCS[f"CLM-{_ref}-notification"] = f"""
MOTOR CLAIM NOTIFICATION
Claim reference: CLM-2026-{_ref}
Policy number: MOT-{_ref}901
Vehicle registration: {_reg}
Pre-accident market value: GBP {_mv}
Incident date: {_date}
Description: {_desc}
"""
    CASE_DOCS[f"CLM-{_ref}-schedule"] = f"""
POLICY SCHEDULE EXTRACT
Policy number: MOT-{_ref}901
Sum insured: GBP {_mv}
Standard excess: GBP {_exc}
Cover type: Comprehensive
"""
    CASE_DOCS[f"CLM-{_ref}-estimate"] = f"""
REPAIR ESTIMATE
Repairer: Northgate Motor Repairs Ltd
Vehicle registration: {_reg}
  Parts and labour, itemised on attached sheet
  TOTAL ..................................... GBP {_total}
"""


# --------------------------------------------------------------------------
# EVALUATION SET
# Each question names the chunks that MUST be retrieved for a correct answer.
# We record them by the source text that must appear, so ground truth stays
# valid across all chunk sizes.
# --------------------------------------------------------------------------
EVAL_SET: list[dict] = [
    {
        "id": "Q01", "type": "single_fact",
        "question": "What is the standard excess on claim CLM-2026-4471?",
        "answer": "GBP 350.00",
        "must_contain": ["Standard excess", "350"],
        "corpus": "case",
    },
    {
        "id": "Q02", "type": "single_fact",
        "question": "What is the total on the original repair estimate for claim CLM-2026-4471, vehicle KP19 TRX?",
        "answer": "GBP 12,400.00",
        "must_contain": ["TOTAL", "12,400"],
        "corpus": "case",
    },
    {
        "id": "Q03", "type": "single_fact",
        "question": "What is the pre-accident market value of vehicle KP19 TRX on claim CLM-2026-4471?",
        "answer": "GBP 8,400.00",
        "must_contain": ["market value", "8,400"],
        "corpus": "case",
    },
    {
        "id": "Q04", "type": "clause_lookup",
        "question": "What does clause 4.1 say about total loss?",
        "answer": "Repair cost above sixty per cent of pre-accident market value.",
        "must_contain": ["4.1", "sixty per cent"],
        "corpus": "policy",
    },
    {
        "id": "Q05", "type": "clause_lookup",
        "question": "What is the rule on repair authorisation above GBP 5,000?",
        "answer": "Clause 3.2 — handler authorisation required before work begins.",
        "must_contain": ["3.2", "authorisation"],
        "corpus": "policy",
    },
    {
        "id": "Q06", "type": "multi_part",
        "question": "Is this claim a total loss, and does the supplementary work "
                    "need an engineer inspection?",
        "answer": "Repair 14,750 against market value 8,400 is 176%, so total loss "
                  "under clause 4.1. Supplementary is 18.9%, below the twenty per "
                  "cent trigger in clause 3.5, so no inspection required on that test.",
        "must_contain": ["4.1", "3.5", "14,750", "8,400"],
        "corpus": "both",
    },
    {
        "id": "Q07", "type": "multi_part",
        "question": "Was the claim notified in time, and was the young driver "
                    "excess applicable?",
        "answer": "Notified 28 days after the incident, inside the thirty-day window "
                  "in clause 6.2. Driver aged 23, so the under-25 young driver excess "
                  "in the 2026 clause 2.4 applies.",
        "must_contain": ["6.2", "2.4", "23", "thirty days"],
        "corpus": "both",
    },
    {
        "id": "Q08", "type": "multi_part",
        "question": "What is the total claim exposure including the supplementary "
                    "estimate, and how does it compare to the sum insured?",
        "answer": "Revised total 14,750 against sum insured 9,000 and market value "
                  "8,400. Clause 2.1 caps payment at the lower of the two.",
        "must_contain": ["14,750", "9,000", "2.1"],
        "corpus": "both",
    },
    {
        "id": "Q09", "type": "version_sensitive",
        "question": "Under the 2026 policy, at what driver age does the young "
                    "driver excess apply, and how much is it?",
        "answer": "Under twenty-five, GBP 250 (2026 wording of clause 2.4).",
        "must_contain": ["twenty-five", "250"],
        "corpus": "policy",
        "trap": "The 2025 version says under twenty-one and GBP 150. "
                "Retrieving the wrong version gives a confident wrong answer.",
    },
    {
        "id": "Q10", "type": "version_sensitive",
        "question": "Under the 2026 policy, what percentage triggers a total loss?",
        "answer": "Sixty per cent (2026 clause 4.1).",
        "must_contain": ["sixty per cent"],
        "corpus": "policy",
        "trap": "The 2025 version says seventy per cent.",
    },
    {
        "id": "Q11", "type": "negative",
        "question": "Has the third party admitted liability on claim CLM-2026-4471?",
        "answer": "not_found",
        "must_contain": [],
        "corpus": "case",
        "expect_no_answer": True,
    },
    {
        "id": "Q12", "type": "negative",
        "question": "What is the courtesy car entitlement under this policy?",
        "answer": "not_found",
        "must_contain": [],
        "corpus": "policy",
        "expect_no_answer": True,
    },
]


def all_docs() -> dict[str, tuple[str, str]]:
    """name -> (corpus, text)"""
    out = {}
    for k, v in POLICY_DOCS.items():
        out[k] = ("policy", v.strip())
    for k, v in CASE_DOCS.items():
        out[k] = ("case", v.strip())
    return out
